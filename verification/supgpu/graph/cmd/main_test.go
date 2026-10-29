package main

import "testing"

func TestStrictModeArguments(t *testing.T) {
	for _, args := range [][]string{nil, {"-mode", "source"}, {"-mode", "consumer", "-admission", "x"}, {"-mode", "consumer", "-admission", "x", "-admission-sha", "y", "position"}, {"-mode", "consumer", "-admission", "x", "-admission-sha", "y", "-request", "x"}, {"-mode", "guarded-child", "-request", "x", "-request-sha", "y", "-mode", "consumer"}, {"-mode", "replay", "-admission", "x", "-admission-sha", "y"}, {"-mode", "consumer", "-unknown", "x"}} {
		if _, e := parse(args); e == nil {
			t.Fatalf("accepted %q", args)
		}
	}
	for _, args := range [][]string{{"-mode", "consumer", "-admission", "x", "-admission-sha", "y"}, {"-mode", "guarded-child", "-request", "x", "-request-sha", "y"}, {"-mode", "replay", "-admission", "x", "-admission-sha", "y", "-receipt", "r", "-receipt-sha", "s"}} {
		if _, e := parse(args); e != nil {
			t.Fatalf("%q: %v", args, e)
		}
	}
}
