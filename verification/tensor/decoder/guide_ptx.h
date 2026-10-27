#pragma once
// Arithmetic: pinned decoder_lib.sm_120.ptx, back_guide_step lines 127-416.
// This entry computes one frame. Go owns terminal initialization and time order.
inline constexpr char decoder_guide_ptx[] = R"ptx(
.version 9.0
.target sm_120
.address_size 64
.visible .entry go_decoder_guide_frame(
 .param .u64 scores,
 .param .u64 next,
 .param .u64 output,
 .param .u64 score_batch_stride,
 .param .u32 states,
 .param .f32 clamp,
 .param .f32 blank
) {
 .reg .pred %p<4>;
 .reg .b16 %h;
 .reg .b32 %r<12>;
 .reg .b64 %a<12>;
 .reg .f32 %f<48>;
 ld.param.u64 %a0,[scores];
 ld.param.u64 %a1,[next];
 ld.param.u64 %a2,[output];
 ld.param.u64 %a3,[score_batch_stride];
 ld.param.u32 %r0,[states];
 ld.param.f32 %f0,[clamp];
 ld.param.f32 %f1,[blank];
 mov.u32 %r1,%ctaid.x;
 mov.u32 %r2,%tid.x;
 mov.u32 %r3,%ntid.x;
 mul.wide.u32 %a4,%r1,2;
 mul.lo.u64 %a5,%a4,%a3;
 add.u64 %a0,%a0,%a5;
 mul.wide.u32 %a4,%r1,4;
 cvt.u64.u32 %a5,%r0;
 mul.lo.u64 %a4,%a4,%a5;
 add.u64 %a1,%a1,%a4;
 add.u64 %a2,%a2,%a4;
 setp.eq.ftz.f32 %p0,%f0,0f00000000;
 selp.f32 %f0,0f7F7FFFFF,%f0,%p0;
 neg.ftz.f32 %f2,%f0;
loop:
 setp.ge.u32 %p1,%r2,%r0;
 @%p1 bra done;
 shl.b32 %r4,%r2,2;
 div.u32 %r5,%r4,%r0;
 mul.lo.u32 %r6,%r5,%r0;
 sub.u32 %r6,%r4,%r6;
 mul.wide.u32 %a4,%r2,4;
 add.u64 %a5,%a1,%a4;
 ld.global.f32 %f3,[%a5];
 add.ftz.f32 %f4,%f1,%f3;
 mul.wide.u32 %a5,%r6,4;
 add.u64 %a5,%a1,%a5;
 ld.global.f32 %f5,[%a5];
 ld.global.f32 %f6,[%a5+4];
 ld.global.f32 %f7,[%a5+8];
 ld.global.f32 %f8,[%a5+12];
 shl.b32 %r7,%r6,2;
 add.u32 %r7,%r7,%r5;
 mul.wide.u32 %a6,%r7,2;
 add.u64 %a6,%a0,%a6;
 ld.global.nc.u16 %h,[%a6];
 cvt.f32.f16 %f9,%h;
 max.ftz.f32 %f9,%f2,%f9;
 min.ftz.f32 %f9,%f0,%f9;
 add.ftz.f32 %f10,%f5,%f9;
 ld.global.nc.u16 %h,[%a6+8];
 cvt.f32.f16 %f9,%h;
 max.ftz.f32 %f9,%f2,%f9;
 min.ftz.f32 %f9,%f0,%f9;
 add.ftz.f32 %f11,%f6,%f9;
 ld.global.nc.u16 %h,[%a6+16];
 cvt.f32.f16 %f9,%h;
 max.ftz.f32 %f9,%f2,%f9;
 min.ftz.f32 %f9,%f0,%f9;
 add.ftz.f32 %f12,%f7,%f9;
 ld.global.nc.u16 %h,[%a6+24];
 cvt.f32.f16 %f9,%h;
 max.ftz.f32 %f9,%f2,%f9;
 min.ftz.f32 %f9,%f0,%f9;
 add.ftz.f32 %f13,%f8,%f9;
 max.ftz.f32 %f14,%f4,%f10;
 max.ftz.f32 %f14,%f14,%f11;
 max.ftz.f32 %f14,%f14,%f12;
 max.ftz.f32 %f14,%f14,%f13;
 sub.ftz.f32 %f15,%f4,%f14;
 mul.ftz.f32 %f15,%f15,0f3FB8AA3B;
 ex2.approx.ftz.f32 %f15,%f15;
 add.ftz.f32 %f16,%f15,0f00000000;
 sub.ftz.f32 %f15,%f10,%f14;
 mul.ftz.f32 %f15,%f15,0f3FB8AA3B;
 ex2.approx.ftz.f32 %f15,%f15;
 add.ftz.f32 %f16,%f16,%f15;
 sub.ftz.f32 %f15,%f11,%f14;
 mul.ftz.f32 %f15,%f15,0f3FB8AA3B;
 ex2.approx.ftz.f32 %f15,%f15;
 add.ftz.f32 %f16,%f16,%f15;
 sub.ftz.f32 %f15,%f12,%f14;
 mul.ftz.f32 %f15,%f15,0f3FB8AA3B;
 ex2.approx.ftz.f32 %f15,%f15;
 add.ftz.f32 %f16,%f16,%f15;
 sub.ftz.f32 %f15,%f13,%f14;
 mul.ftz.f32 %f15,%f15,0f3FB8AA3B;
 ex2.approx.ftz.f32 %f15,%f15;
 add.ftz.f32 %f16,%f16,%f15;
 lg2.approx.ftz.f32 %f17,%f16;
 fma.rn.ftz.f32 %f18,%f17,0f3F317218,%f14;
 add.u64 %a7,%a2,%a4;
 st.global.f32 [%a7],%f18;
 add.u32 %r2,%r2,%r3;
 bra loop;
done:
 ret;
}
)ptx";
