module {
  // expected-error@+1 {{unregistered operation 'mini.unknown' found in dialect}}
  "mini.unknown"() : () -> ()
}
