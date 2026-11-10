
#ifndef MINI_OPS_TD
#define MINI_OPS_TD

include "Mini/MiniDialect.td"

def PingOp : Op<Mini_Dialect, "ping"> {
  let summary = "Minimal operation for testing the Mini dialect";

  let description = [{
    A minimal operation used to verify dialect registration,
    TableGen generation, parsing, and printing.
  }];

  let arguments = (ins);
  let results = (outs);
}

#endif // MINI_OPS_TD
