#pragma once
// Numeric instructions follow selected decoder_lib.sm_120.ptx SHA256
// 7e46f59645025bca2cf7a519c9f0334b5435942ca9006d0dec5faa6d3e63b01a.
// Go supplies selected half values, ordered numeric group rows and counters.
inline constexpr char decoder_quality_ptx[] = R"ptx(
.version 9.0
.target sm_120
.address_size 64
.visible .entry go_decoder_quality_numbers(
 .param .u64 input_a,.param .u64 input_b,.param .u64 output,.param .u64 auxiliary,
 .param .u32 rows,.param .u32 width,.param .u32 operation,
 .param .f32 scale,.param .f32 shift
) {
 .reg .pred %p<8>;
 .reg .b16 %h;
 .reg .b32 %r<16>;
 .reg .b64 %a<12>;
 .reg .f32 %f<24>;
 ld.param.u64 %a0,[input_a];
 ld.param.u64 %a1,[input_b];
 ld.param.u64 %a2,[output];
 ld.param.u64 %a3,[auxiliary];
 ld.param.u32 %r0,[rows];
 ld.param.u32 %r1,[width];
 ld.param.u32 %r2,[operation];
 ld.param.f32 %f20,[scale];
 ld.param.f32 %f21,[shift];
 mov.u32 %r3,%ctaid.x;
 mov.u32 %r4,%ntid.x;
 mov.u32 %r5,%tid.x;
 mad.lo.u32 %r3,%r3,%r4,%r5;
 setp.ge.u32 %p0,%r3,%r0;
 @%p0 bra done;
 setp.eq.u32 %p1,%r2,5;
 @%p1 bra summary_addresses;
 mul.wide.u32 %a4,%r3,4;
 add.u64 %a1,%a1,%a4;
 add.u64 %a2,%a2,%a4;
 setp.eq.u32 %p1,%r2,3;
 @%p1 bra probability;
 mul.wide.u32 %a4,%r3,4;
 cvt.u64.u32 %a5,%r1;
 mul.lo.u64 %a4,%a4,%a5;
 add.u64 %a0,%a0,%a4;
 cvt.u64.u32 %a6,%r3;
 add.u64 %a3,%a3,%a6;
 ld.global.u32 %r6,[%a1];
 setp.ne.u32 %p0,%r6,%r1;
 @%p0 bra done;
 mov.f32 %f0,0f00000000;
 mov.u32 %r7,0;
group_sum:
 ld.global.f32 %f1,[%a0];
 add.ftz.f32 %f0,%f0,%f1;
 add.u64 %a0,%a0,4;
 add.u32 %r7,%r7,1;
 setp.lt.u32 %p0,%r7,%r6;
 @%p0 bra group_sum;
 cvt.rn.f32.s32 %f2,%r6;
 div.approx.ftz.f32 %f3,%f0,%f2;
 sub.ftz.f32 %f4,0f3F800000,%f3;
 lg2.approx.ftz.f32 %f5,%f4;
 mul.ftz.f32 %f6,%f5,0fBE9A209B;
 mul.ftz.f32 %f7,%f6,0f41200000;
 fma.rn.ftz.f32 %f8,%f20,%f7,%f21;
 min.ftz.f32 %f9,%f8,0f42480000;
 max.ftz.f32 %f10,%f9,0f3F800000;
 add.ftz.f32 %f11,%f10,0f42060000;
 cvt.rzi.ftz.s32.f32 %r8,%f11;
 st.global.f32 [%a2],%f4;
 st.global.u8 [%a3],%r8;
 bra done;
probability:
 mul.wide.u32 %a4,%r3,18;
 add.u64 %a0,%a0,%a4;
 ld.global.u32 %r6,[%a1];
 setp.gt.u32 %p0,%r6,8;
 @%p0 bra done;
 ld.global.u16 %h,[%a0+16];
 cvt.f32.f16 %f2,%h;
 mov.f32 %f0,0f00000000;
 mov.u32 %r7,0;
 setp.eq.u32 %p0,%r6,0;
 @%p0 bra probability_center;
probability_sum:
 ld.global.u16 %h,[%a0];
 cvt.f32.f16 %f1,%h;
 add.ftz.f32 %f0,%f0,%f1;
 add.u64 %a0,%a0,2;
 add.u32 %r7,%r7,1;
 setp.lt.u32 %p0,%r7,%r6;
 @%p0 bra probability_sum;
probability_center:
 add.ftz.f32 %f3,%f2,%f0;
 max.ftz.f32 %f4,%f3,0f00000000;
 min.ftz.f32 %f5,%f4,0f3F800000;
 lg2.approx.ftz.f32 %f6,%f5;
 mul.ftz.f32 %f7,%f6,0f3ECCCCCD;
 ex2.approx.ftz.f32 %f8,%f7;
 st.global.f32 [%a2],%f8;
 bra done;
summary_addresses:
 mul.wide.u32 %a4,%r3,8;
 add.u64 %a1,%a1,%a4;
 mul.wide.u32 %a4,%r3,12;
 add.u64 %a2,%a2,%a4;
 cvt.u64.u32 %a4,%r3;
 add.u64 %a3,%a3,%a4;
 mul.wide.u32 %a4,%r3,4;
 cvt.u64.u32 %a5,%r1;
 mul.lo.u64 %a4,%a4,%a5;
 add.u64 %a0,%a0,%a4;
 ld.global.u32 %r6,[%a1];
 ld.global.u32 %r8,[%a1+4];
 setp.eq.u32 %p0,%r6,0;
 setp.gt.u32 %p1,%r6,2147483647;
 or.pred %p0,%p0,%p1;
 setp.gt.u32 %p1,%r1,%r6;
 or.pred %p0,%p0,%p1;
 setp.gt.u32 %p1,%r8,%r6;
 or.pred %p0,%p0,%p1;
 @%p0 bra done;
 mov.f32 %f0,0f00000000;
 mov.u32 %r7,0;
summary_sum:
 ld.global.f32 %f1,[%a0];
 add.ftz.f32 %f0,%f0,%f1;
 add.u64 %a0,%a0,4;
 add.u32 %r7,%r7,1;
 setp.lt.u32 %p0,%r7,%r1;
 @%p0 bra summary_sum;
 cvt.rn.f32.s32 %f2,%r6;
 rcp.approx.ftz.f32 %f3,%f2;
 sub.u32 %r9,%r6,%r8;
 cvt.rn.f32.u32 %f4,%r9;
 mul.ftz.f32 %f5,%f3,%f4;
 cvt.rn.f32.u32 %f6,%r8;
 mul.ftz.f32 %f7,%f3,%f6;
 div.approx.ftz.f32 %f8,%f0,%f6;
 st.global.f32 [%a2],%f5;
 st.global.f32 [%a2+4],%f7;
 st.global.f32 [%a2+8],%f8;
 mov.u32 %r10,1;
 st.global.u8 [%a3],%r10;
done:
 ret;
}
)ptx";
