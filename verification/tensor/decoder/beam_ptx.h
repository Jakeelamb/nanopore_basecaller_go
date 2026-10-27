#pragma once
// Pinned decoder_lib.sm_120.ptx SHA256 7e46f59645025bca2cf7a519c9f0334b5435942ca9006d0dec5faa6d3e63b01a.
// Numeric-only extraction. Log-add retains original lines 1608-1650 verbatim,
// except for its local branch label. Go selects all operands and operations.
inline constexpr char decoder_beam_ptx[] = R"ptx(
.version 9.0
.target sm_120
.address_size 64
.visible .entry go_decoder_beam_numbers(
 .param .u64 input_a,
 .param .u64 input_b,
 .param .u64 input_c,
 .param .u64 output,
 .param .u32 rows,
 .param .u32 operation,
 .param .f32 clamp,
 .param .f32 blank
) {
 .reg .pred %p<100>;
 .reg .b16 %h;
 .reg .b32 %r<550>;
 .reg .b64 %a<12>;
 .reg .f32 %f<190>;
 ld.param.u64 %a0,[input_a];
 ld.param.u64 %a1,[input_b];
 ld.param.u64 %a2,[input_c];
 ld.param.u64 %a3,[output];
 ld.param.u32 %r0,[rows];
 ld.param.u32 %r1,[operation];
 ld.param.f32 %f90,[clamp];
 ld.param.f32 %f91,[blank];
 mov.u32 %r2,%ctaid.x;
 mov.u32 %r3,%ntid.x;
 mov.u32 %r4,%tid.x;
 mad.lo.u32 %r2,%r2,%r3,%r4;
 setp.ge.u32 %p0,%r2,%r0;
 @%p0 bra done;
 setp.eq.u32 %p1,%r1,2;
 @%p1 bra expand;
 mul.wide.u32 %a4,%r2,4;
 add.u64 %a3,%a3,%a4;
 setp.eq.u32 %p1,%r1,1;
 @%p1 bra log_cut;
 mul.wide.u32 %a4,%r2,8;
 add.u64 %a0,%a0,%a4;
 ld.global.f32 %f16,[%a0];
 ld.global.f32 %f17,[%a0+4];
 setp.eq.u32 %p1,%r1,3;
 @%p1 bra log_add;
 setp.eq.u32 %p1,%r1,4;
 @%p1 bra subtract;
 setp.eq.u32 %p1,%r1,5;
 @%p1 bra midpoint;
 max.ftz.f32 %f0,%f16,%f17;
 bra store;
subtract:
 sub.ftz.f32 %f0,%f16,%f17;
 bra store;
midpoint:
 add.ftz.f32 %f0,%f16,%f17;
 div.approx.ftz.f32 %f0,%f0,0f40000000;
 bra store;
log_cut:
 add.u64 %a0,%a0,%a4;
 ld.global.f32 %f0,[%a0];
 lg2.approx.ftz.f32 %f0,%f0;
 mul.ftz.f32 %f0,%f0,0f3F317218;
 bra store;
log_add:
sub.ftz.f32 %f136, %f16, %f17;
abs.ftz.f32 %f18, %f136;
max.ftz.f32 %f19, %f16, %f17;
setp.geu.ftz.f32 %p93, %f18, 0f41880000;
mov.f32 %f185, 0f00000000;
@%p93 bra beam_logadd_tail;
mul.ftz.f32 %f137, %f18, 0fBFB8AA3B;
ex2.approx.ftz.f32 %f20, %f137;
mov.f32 %f138, 0f3F800000;
add.rz.ftz.f32 %f139, %f20, %f138;
mov.b32 %r540, %f139;
add.s32 %r541, %r540, -1061158912;
and.b32 %r542, %r541, -8388608;
mov.b32 %r101, %f20;
sub.s32 %r543, %r101, %r542;
mov.b32 %f140, %r543;
sub.s32 %r544, 1082130432, %r542;
mov.b32 %f141, %r544;
fma.rn.ftz.f32 %f142, %f141, 0f3E800000, 0fBF800000;
add.ftz.f32 %f143, %f142, %f140;
cvt.rn.f32.s32 %f144, %r542;
mul.ftz.f32 %f145, %f144, 0f34000000;
fma.rn.ftz.f32 %f146, %f143, 0fBD39BF78, 0f3DD80012;
fma.rn.ftz.f32 %f147, %f146, %f143, 0fBE0778E0;
fma.rn.ftz.f32 %f148, %f147, %f143, 0f3E146475;
fma.rn.ftz.f32 %f149, %f148, %f143, 0fBE2A68DD;
fma.rn.ftz.f32 %f150, %f149, %f143, 0f3E4CAF9E;
fma.rn.ftz.f32 %f151, %f150, %f143, 0fBE800042;
fma.rn.ftz.f32 %f152, %f151, %f143, 0f3EAAAAE6;
fma.rn.ftz.f32 %f153, %f152, %f143, 0fBF000000;
mul.ftz.f32 %f154, %f143, %f153;
fma.rn.ftz.f32 %f155, %f154, %f143, %f143;
fma.rn.ftz.f32 %f185, %f145, 0f3F317218, %f155;
setp.lt.u32 %p94, %r101, 2139095040;
@%p94 bra beam_logadd_tail;
setp.gt.s32 %p95, %r101, -1082130432;
fma.rn.ftz.f32 %f156, %f20, 0f7F800000, 0f7F800000;
selp.f32 %f157, %f156, %f185, %p95;
setp.eq.ftz.f32 %p96, %f20, 0f00000000;
selp.f32 %f185, 0f80000000, %f157, %p96;
beam_logadd_tail:
add.ftz.f32 %f158, %f19, %f185;
 mov.f32 %f0,%f158;
store:
 st.global.f32 [%a3],%f0;
 bra done;
expand:
 mul.wide.u32 %a4,%r2,4;
 add.u64 %a0,%a0,%a4;
 mul.wide.u32 %a4,%r2,8;
 add.u64 %a1,%a1,%a4;
 mul.wide.u32 %a4,%r2,20;
 add.u64 %a2,%a2,%a4;
 mul.wide.u32 %a4,%r2,24;
 add.u64 %a3,%a3,%a4;
 setp.eq.ftz.f32 %p0,%f90,0f00000000;
 selp.f32 %f90,0f7F7FFFFF,%f90,%p0;
 neg.ftz.f32 %f92,%f90;
 ld.global.f32 %f0,[%a0];
 ld.global.f32 %f1,[%a2];
 ld.global.f32 %f2,[%a2+4];
 ld.global.f32 %f3,[%a2+8];
 ld.global.f32 %f4,[%a2+12];
 ld.global.f32 %f5,[%a2+16];
 ld.global.u16 %h,[%a1+0];
 cvt.f32.f16 %f6,%h;
 max.ftz.f32 %f6,%f92,%f6;
 min.ftz.f32 %f6,%f90,%f6;
 add.ftz.f32 %f6,%f0,%f6;
 add.ftz.f32 %f10,%f1,%f6;
 ld.global.u16 %h,[%a1+2];
 cvt.f32.f16 %f6,%h;
 max.ftz.f32 %f6,%f92,%f6;
 min.ftz.f32 %f6,%f90,%f6;
 add.ftz.f32 %f6,%f0,%f6;
 add.ftz.f32 %f11,%f2,%f6;
 ld.global.u16 %h,[%a1+4];
 cvt.f32.f16 %f6,%h;
 max.ftz.f32 %f6,%f92,%f6;
 min.ftz.f32 %f6,%f90,%f6;
 add.ftz.f32 %f6,%f0,%f6;
 add.ftz.f32 %f12,%f3,%f6;
 ld.global.u16 %h,[%a1+6];
 cvt.f32.f16 %f6,%h;
 max.ftz.f32 %f6,%f92,%f6;
 min.ftz.f32 %f6,%f90,%f6;
 add.ftz.f32 %f6,%f0,%f6;
 add.ftz.f32 %f13,%f4,%f6;
 add.ftz.f32 %f14,%f91,%f0;
 add.ftz.f32 %f14,%f14,%f5;
 max.ftz.f32 %f15,0fFF7FFFFF,%f10;
 max.ftz.f32 %f15,%f15,%f11;
 max.ftz.f32 %f15,%f15,%f12;
 max.ftz.f32 %f15,%f15,%f13;
 max.ftz.f32 %f15,%f15,%f14;
 st.global.f32 [%a3],%f10;
 st.global.f32 [%a3+4],%f11;
 st.global.f32 [%a3+8],%f12;
 st.global.f32 [%a3+12],%f13;
 st.global.f32 [%a3+16],%f14;
 st.global.f32 [%a3+20],%f15;
done:
 ret;
}
)ptx";
