#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "nm_config.hpp"

using namespace mlir;
namespace {
struct PrepareExecutors : PassWrapper<PrepareExecutors, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareExecutors)
  StringRef getArgument() const final { return "nm-prepare-executors"; }
  StringRef getDescription() const final {
    return "Validate packed-W4 executor ABI and generate a calibrated-risk predicate interface";
  }
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect>();
  }
  bool checkFunction(ModuleOp module, StringRef symbol, unsigned id, bool a8) {
    auto f = module.lookupSymbol<func::FuncOp>(symbol);
    if (!f || f.isExternal()) { module.emitError("missing numerical executor: ") << symbol; return false; }
    auto type = f.getFunctionType();
    auto variant = f->getAttrOfType<IntegerAttr>("nm.variant_id");
    auto format = f->getAttrOfType<StringAttr>("nm.weight_format");
    if (!variant || variant.getInt() != id || !format || format.getValue() != "packed_w4" ||
        type.getNumInputs() != (a8 ? 9U : 8U) || type.getNumResults() != 0) {
      f.emitError("incorrect variant metadata or executor signature"); return false;
    }
    for (unsigned i = 0; i < 4; ++i) {
      auto memref = dyn_cast<MemRefType>(type.getInput(i));
      if (!memref || memref.getRank() != 1 || !memref.isDynamicDim(0) || !memref.getLayout().isIdentity()) {
        f.emitError("executor ABI requires rank-one dynamic identity-layout memrefs"); return false;
      }
      auto element = memref.getElementType();
      const bool correct = i == 0 ? element.isInteger(a8 ? 8 : 16) : i == 1 ? element.isInteger(8) : element.isF32();
      if (!correct) { f.emitError("wrong executor buffer element type"); return false; }
    }
    for (unsigned i = 4; i < 8; ++i) if (!type.getInput(i).isIndex()) {
      f.emitError("M/N/K/G must have index type"); return false;
    }
    if (a8 && !type.getInput(8).isF32()) { f.emitError("A8 activation scale must be f32"); return false; }
    f->setAttr("nm.approved", UnitAttr::get(&getContext()));
    f->setAttr("nm.configuration_version", IntegerAttr::get(IntegerType::get(&getContext(), 64), nm::generated::config_version));
    return true;
  }
  void runOnOperation() override {
    auto module = getOperation();
    if (!checkFunction(module, "nm_bf16", 1, false) || !checkFunction(module, "nm_a8", 2, true)) {
      signalPassFailure(); return;
    }
    if (module.lookupSymbol("nm_risk_guard")) { module.emitError("risk guard symbol already exists"); signalPassFailure(); return; }
    OpBuilder b(module.getContext());
    b.setInsertionPointToEnd(module.getBody());
    auto location = module.getLoc();
    auto type = b.getFunctionType({b.getF64Type(), b.getF64Type(), b.getF64Type(), b.getF64Type(), b.getI32Type()}, {b.getI32Type()});
    auto guard = b.create<func::FuncOp>(location, "nm_risk_guard", type);
    guard->setAttr("llvm.emit_c_interface", b.getUnitAttr());
    guard->setAttr("nm.configuration_version", b.getI64IntegerAttr(nm::generated::config_version));
    auto* entry = guard.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Value maxOk = b.create<arith::CmpFOp>(location, arith::CmpFPredicate::OLE, entry->getArgument(0), entry->getArgument(2));
    Value clippingOk = b.create<arith::CmpFOp>(location, arith::CmpFPredicate::OLE, entry->getArgument(1), entry->getArgument(3));
    Value zero = b.create<arith::ConstantIntOp>(location, 0, 32);
    Value enabled = b.create<arith::CmpIOp>(location, arith::CmpIPredicate::ne, entry->getArgument(4), zero);
    Value accepted = b.create<arith::AndIOp>(location, maxOk, clippingOk);
    accepted = b.create<arith::AndIOp>(location, accepted, enabled);
    Value result = b.create<arith::ExtUIOp>(location, b.getI32Type(), accepted);
    b.create<func::ReturnOp>(location, result);
  }
};
}
void registerNeuroMindsPasses() { PassRegistration<PrepareExecutors>(); }
