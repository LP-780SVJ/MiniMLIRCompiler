module {
  func.func @unsupported() {
    "mini.ping"() : () -> ()
    func.return
  }
}
