#pragma once
#define D01_FAKE_ARGS void*,void*,void*,int,void*,float,int,void*,void*,void*,void*,void*,void*,float,float,int,float,float
int host_back_guide_step(D01_FAKE_ARGS);
int host_beam_search_step(D01_FAKE_ARGS);
int host_compute_posts_step(D01_FAKE_ARGS);
int host_run_decode(D01_FAKE_ARGS,int);
