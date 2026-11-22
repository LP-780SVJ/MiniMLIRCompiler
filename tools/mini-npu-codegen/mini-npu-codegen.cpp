#include "Mini/MiniDialect.h"
#include "NPU/NPUDialect.h"
#include "NPU/ProgramPackage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <cerrno>
#include <filesystem>
#if defined(__APPLE__)
#include <cstdio>
#elif defined(__linux__)
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

using namespace mlir;
using namespace mlir::npu;
namespace fs = std::filesystem;
namespace cl = llvm::cl;

namespace {

cl::opt<std::string> input("input", cl::desc("Scheduled MLIR file"), cl::Required);
cl::opt<std::string> target("target", cl::desc("Virtual NPU target JSON"), cl::Required);
cl::opt<std::string> metadata("model-metadata", cl::desc("Optional ONNX ABI metadata"));
cl::opt<std::string> outDir("out-dir", cl::desc("New program package directory"), cl::Required);

llvm::Error writeFile(StringRef path, StringRef bytes) {
  std::error_code ec;
  llvm::raw_fd_ostream file(path, ec, llvm::sys::fs::OF_None);
  if (ec)
    return llvm::createStringError(ec, "cannot create %s", path.str().c_str());
  file << bytes;
  file.close();
  if (file.has_error()) {
    ec = file.error();
    file.clear_error();
    return llvm::createStringError(ec, "cannot write %s", path.str().c_str());
  }
  return llvm::Error::success();
}

llvm::Error validateStaging(StringRef executable, StringRef directory) {
  auto python = llvm::sys::findProgramByName("python3");
  if (!python)
    return llvm::createStringError(python.getError(), "cannot find python3 for package validation");
  SmallString<256> script(llvm::sys::path::parent_path(executable));
  llvm::sys::path::append(script, "mini-npu-validate-package.py");
  std::string message;
  int status = llvm::sys::ExecuteAndWait(*python,
      {*python, "-I", script, "--package", directory, "--quiet"}, std::nullopt,
      {}, 0, 0, &message);
  if (status != 0)
    return llvm::createStringError("staged package validation failed (exit %d)%s%s",
                                   status, message.empty() ? "" : ": ", message.c_str());
  return llvm::Error::success();
}

llvm::Error renameExclusive(const char *staging, const char *destination) {
#if defined(__APPLE__)
  if (::renamex_np(staging, destination, RENAME_EXCL) == 0)
    return llvm::Error::success();
#elif defined(__linux__) && defined(SYS_renameat2)
  if (::syscall(SYS_renameat2, AT_FDCWD, staging, AT_FDCWD, destination,
                RENAME_NOREPLACE) == 0)
    return llvm::Error::success();
#else
  return llvm::createStringError("atomic no-replace publication is unavailable on this platform");
#endif
  return llvm::createStringError(std::error_code(errno, std::generic_category()),
                                 "cannot publish package to %s", destination);
}

llvm::Error publish(const ProgramPackage &package, StringRef executable) {
  fs::path destination(outDir.getValue());
  if (!destination.empty() && !destination.has_filename())
    destination = destination.parent_path();
  if (destination.empty() || !destination.has_filename() ||
      destination.filename() == "." || destination.filename() == "..")
    return llvm::createStringError("out-dir must name a new package directory");
  std::error_code ec;
  const auto type = fs::symlink_status(destination, ec).type();
  if (type != fs::file_type::not_found) {
    if (ec)
      return llvm::createStringError(ec, "cannot inspect out-dir");
    return llvm::createStringError("out-dir already exists: %s", destination.c_str());
  }

  SmallString<256> staging;
  if (auto error = llvm::sys::fs::createUniqueDirectory(
          destination.string() + ".tmp", staging))
    return llvm::createStringError(error, "cannot create package staging directory");
  auto result = [&]() -> llvm::Error {
    for (auto [name, bytes] : {std::pair<StringRef, StringRef>{"program.json", package.program},
                              {"weights.bin", package.weights},
                              {"manifest.json", package.manifest}}) {
      SmallString<256> path(staging);
      llvm::sys::path::append(path, name);
      if (auto error = writeFile(path, bytes))
        return error;
    }
    if (auto error = validateStaging(executable, staging))
      return error;
    return renameExclusive(staging.c_str(), destination.c_str());
  }();
  if (result) {
    if (auto error = llvm::sys::fs::remove_directories(staging))
      return llvm::joinErrors(std::move(result),
                              llvm::createStringError(error, "cannot clean staging directory"));
  }
  return result;
}

llvm::Error generate(StringRef executable) {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, mini::MiniDialect, NPUDialect>();
  MLIRContext context(registry);
  ParserConfig parser(&context, /*verifyAfterParse=*/false);
  auto module = parseSourceFile<ModuleOp>(input.getValue(), parser);
  if (!module || failed(verify(*module)))
    return llvm::createStringError("scheduled MLIR parsing or verification failed");

  auto profile = llvm::MemoryBuffer::getFile(target.getValue(), /*IsText=*/false,
      /*RequiresNullTerminator=*/true, /*IsVolatile=*/true);
  if (!profile)
    return llvm::createStringError(profile.getError(), "cannot read target profile");
  auto package = buildProgramPackage(*module, (*profile)->getBuffer(), metadata.getValue());
  if (!package)
    return package.takeError();
  return publish(*package, executable);
}

}

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "Export verified Virtual NPU ISA v1 packages\n");
  static int executableAnchor;
  auto executable = llvm::sys::fs::getMainExecutable(argv[0], &executableAnchor);
  if (auto error = generate(executable)) {
    llvm::logAllUnhandledErrors(std::move(error), llvm::errs(), "mini-npu-codegen: ");
    return 1;
  }
  return 0;
}
