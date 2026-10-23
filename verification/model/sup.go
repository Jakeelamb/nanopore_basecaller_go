package modeloracle

import _ "embed"

//go:embed native/sup.cpp
var supAdapter string

var supPins = []sourcePin{
	{"dorado/nn/TxModules.cpp", "9e73e81b38c6b6e36a17566a8dfc16e2da2e54fab44dfdbcb85485d77998b701", [][2]int{{128, 400}, {756, 958}, {350, 380}}},
	{"dorado/nn/RMSNorm.cpp", "5c98255bf9942f41f80811455c04a314e244cc925aca5b0bf5886f1e89fcec66", [][2]int{{8, 17}}},
	{"dorado/nn/include/nn/RMSNorm.h", "3a80d3af6756682ec91e17f7335310e2a351b23802541b81b5bfcbae5ab3dba6", nil},
	{"dorado/nn/LinearUpsample.cpp", "a863639ded16c392076b8feb0cf730ce021f2326fb35f3941a6542c8faea7e3c", [][2]int{{7, 23}}},
	{"dorado/nn/ConvStack.cpp", "d71cbdb1100010991753e4b3b80b8866265d97a08855f0b8ff1e41c00c91f6b2", [][2]int{{110, 128}, {169, 188}}},
	{"dorado/config/include/config/common.h", "4d5011d1a9502c18530b48cda4ccc4db2696778bfa0efdcb4758fbbafebb0340", nil},
	{"dorado/config/include/config/BasecallModelConfig.h", "55f5abcc032582d23370ff54c6b7645d1914dc9c458e0f57d56b0d15d23be5d8", [][2]int{{46, 89}}},
	{"dorado/utils/include/utils/math_utils.h", "da7d19142cc0d858a1d954257a0f5f3d204b14ee3e5d1611da6038fc9d931101", [][2]int{{19, 26}}},
	{"dorado/torch_utils/torch_utils.cpp", "01a711350bea009e9c0b5d0a3578327c10ac65fa773c1cec1e1cafafd8b19482", [][2]int{{15, 34}}},
	{"dorado/basecall/model/TxModel.cpp", "fb9184a3755819242c820c24cb9c8976022f88e22908fa2097f2f0a4047a20ed", [][2]int{{20, 48}}},
}

func ExtractSUP(root string) ([]Source, error) { return extract(root, supPins) }

// BuildSUP runs only bounded component and reduced-shape source-function fixtures.
// Stock-weight graph execution is a separate explicit RunStock call.
func BuildSUP(sourceRoot, library, output string) (string, error) {
	return build(sourceRoot, library, output, supAdapter, supPins, true)
}
