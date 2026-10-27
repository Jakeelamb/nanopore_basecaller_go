#pragma once
// Selected SM120 compute_posts_step: ordered lane maxima/sums, butterfly
// reductions, approximate log-normalizer and half stores. One row only;
// initialization, time order and destructive score placement belong to Go.
inline constexpr char decoder_posterior_ptx[] = R"ptx(
.version 9.0
.target sm_120
.address_size 64
.visible .entry go_decoder_posterior_row(
 .param .u64 alpha,
 .param .u64 beta,
 .param .u64 output,
 .param .u64 output_batch_stride,
 .param .u32 states
) {
 .reg .pred %p<4>;
 .reg .b16 %h;
 .reg .b32 %r<12>;
 .reg .b64 %a<10>;
 .reg .f32 %f<16>;
 .shared .align 4 .b8 values[4096];
 .shared .align 4 .f32 normalizer;
 ld.param.u64 %a0,[alpha];
 ld.param.u64 %a1,[beta];
 ld.param.u64 %a2,[output];
 ld.param.u64 %a3,[output_batch_stride];
 ld.param.u32 %r0,[states];
 mov.u32 %r1,%ctaid.x;
 mov.u32 %r2,%tid.x;
 mul.lo.u32 %r3,%r1,%r0;
 add.u32 %r3,%r3,%r2;
 mul.wide.u32 %a4,%r3,4;
 add.u64 %a5,%a0,%a4;
 ld.global.f32 %f0,[%a5];
 add.u64 %a5,%a1,%a4;
 ld.global.f32 %f1,[%a5];
 add.ftz.f32 %f2,%f0,%f1;
 mov.u32 %r4,values;
 shl.b32 %r5,%r2,2;
 add.u32 %r5,%r4,%r5;
 st.shared.f32 [%r5],%f2;
 bar.sync 0;
 setp.ge.u32 %p0,%r2,32;
 @%p0 bra posterior_barrier;
 mov.f32 %f3,0f00000000;
 mov.u32 %r6,%r2;
posterior_max:
 shl.b32 %r7,%r6,2;
 add.u32 %r7,%r4,%r7;
 ld.shared.f32 %f4,[%r7];
 max.ftz.f32 %f3,%f3,%f4;
 add.u32 %r6,%r6,32;
 setp.lt.u32 %p1,%r6,%r0;
 @%p1 bra posterior_max;
 mov.u32 %r8,16;
posterior_max_shuffle:
 mov.b32 %r9,%f3;
 shfl.sync.bfly.b32 %r10|%p2,%r9,%r8,31,-1;
 mov.b32 %f4,%r10;
 max.ftz.f32 %f3,%f3,%f4;
 shr.u32 %r8,%r8,1;
 setp.ne.u32 %p1,%r8,0;
 @%p1 bra posterior_max_shuffle;
 mov.f32 %f5,0f00000000;
 mov.u32 %r6,%r2;
posterior_sum:
 shl.b32 %r7,%r6,2;
 add.u32 %r7,%r4,%r7;
 ld.shared.f32 %f4,[%r7];
 sub.ftz.f32 %f4,%f4,%f3;
 mul.ftz.f32 %f4,%f4,0f3FB8AA3B;
 ex2.approx.ftz.f32 %f4,%f4;
 add.ftz.f32 %f5,%f5,%f4;
 add.u32 %r6,%r6,32;
 setp.lt.u32 %p1,%r6,%r0;
 @%p1 bra posterior_sum;
 mov.u32 %r8,16;
posterior_sum_shuffle:
 mov.b32 %r9,%f5;
 shfl.sync.bfly.b32 %r10|%p2,%r9,%r8,31,-1;
 mov.b32 %f4,%r10;
 add.ftz.f32 %f5,%f5,%f4;
 shr.u32 %r8,%r8,1;
 setp.ne.u32 %p1,%r8,0;
 @%p1 bra posterior_sum_shuffle;
 lg2.approx.ftz.f32 %f6,%f5;
 fma.rn.ftz.f32 %f7,%f6,0f3F317218,%f3;
 setp.ne.u32 %p1,%r2,0;
 @%p1 bra posterior_barrier;
 st.shared.f32 [normalizer],%f7;
posterior_barrier:
 bar.sync 0;
 ld.shared.f32 %f7,[normalizer];
 sub.ftz.f32 %f8,%f2,%f7;
 mul.ftz.f32 %f8,%f8,0f3FB8AA3B;
 ex2.approx.ftz.f32 %f8,%f8;
 cvt.rn.f16.f32 %h,%f8;
 mul.wide.u32 %a6,%r1,2;
 mul.lo.u64 %a6,%a6,%a3;
 mul.wide.u32 %a7,%r2,2;
 add.u64 %a6,%a6,%a7;
 add.u64 %a6,%a2,%a6;
 st.global.u16 [%a6],%h;
 ret;
}
)ptx";
