#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <vector>

extern "C" {
unsigned char* bd_rgb(); unsigned char* bd_mask(); float* bd_depth(); float* bd_rotation();
unsigned char* bd_labels(); float* bd_branches(); int bd_branch_count(); int bd_junction_count();
int bd_fields();
int bd_run(float,float,float,float,int);
}

static void line(int x0,int y0,int x1,int y1,int radius) {
    int steps=std::max(std::abs(x1-x0),std::abs(y1-y0));
    for(int t=0;t<=steps;t++) {
        int x=x0+(x1-x0)*t/steps,y=y0+(y1-y0)*t/steps;
        for(int dy=-radius;dy<=radius;dy++)for(int dx=-radius;dx<=radius;dx++){
            if(x+dx<0||x+dx>=640||y+dy<0||y+dy>=640)continue;
            bd_mask()[(y+dy)*640+x+dx]=255;
        }
    }
}
int main(int argc,char** argv) {
    constexpr int N=640*640;
    auto* rgb=bd_rgb();auto* mask=bd_mask();auto* depth=bd_depth();
    float fx=190,fy=337,cx=320,cy=320;
    int worldUp=2;
    for(int i=0;i<N;i++) {mask[i]=0;depth[i]=.5f+float(i%640)/1280;
        rgb[i*4]=85;rgb[i*4+1]=65;rgb[i*4+2]=40;rgb[i*4+3]=255;
    }
    float* r=bd_rotation();for(int i=0;i<9;i++)r[i]=(i%4==0)?1:0;
    if(argc>1) {
        std::ifstream f(argv[1],std::ios::binary);
        if(!f || !f.read((char*)rgb,N*4) || !f.read((char*)mask,N) || !f.read((char*)depth,N*4)) {
            std::fprintf(stderr,"Fixture must contain RGBA u8, mask u8, depth f32 at 640x640\n");return 2;
        }
        float camera[13];std::int32_t up;
        if(!f.read((char*)camera,sizeof(camera))||!f.read((char*)&up,sizeof(up))) {
            std::fprintf(stderr,"Fixture must also contain 4 intrinsics, 9 rotation floats, and world-up int32\n");return 2;
        }
        fx=camera[0];fy=camera[1];cx=camera[2];cy=camera[3];
        for(int i=0;i<9;i++)r[i]=camera[4+i];
        worldUp=up;
    } else {
        // Camera +Y maps to world +Z. A broad upright stem supports two
        // narrower, ground-parallel branches at one junction.
        const float zUpRotation[9]={1,0,0,0,0,-1,0,1,0};
        for(int i=0;i<9;i++)r[i]=zUpRotation[i];
        line(320,550,320,310,12);
        line(320,310,125,310,4);
        line(320,310,515,310,4);
    }
    // Warm-up removes one-time runtime/cache effects.
    for(int i=0;i<5;i++)bd_run(fx,fy,cx,cy,worldUp);
    std::vector<double> ms;ms.reserve(150);
    for(int i=0;i<150;i++) {
        auto t0=std::chrono::steady_clock::now();
        bd_run(fx,fy,cx,cy,worldUp);
        ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());
    }
    std::sort(ms.begin(),ms.end());
    int branches=bd_branch_count(), junctions=bd_junction_count();
    std::printf("branches=%d junctions=%d p50_ms=%.2f p95_ms=%.2f max_ms=%.2f\n",
        branches,junctions,ms[75],ms[142],ms.back());
    for(int i=0;i<std::min(branches,15);i++) {
        float* b=bd_branches()+i*bd_fields();
        std::printf("branch %d score=%.3f length=%.0f area=%.0f width=%.1f vertical=%.2f confidence=%.2f trunk=%d imageUp=%.2f\n",
            int(b[0]),b[1],b[2],b[3],b[4],b[20],b[21],int(b[22]),b[23]);
    }
    if(argc==1) {
        bool trunk=false,horizontal=false;
        for(int i=0;i<branches;i++){
            float* b=bd_branches()+i*bd_fields();
            trunk|=b[22]>0.5f;
            horizontal|=b[20]<0.423f&&b[21]>=0.5f&&b[2]>=16&&b[2]/(b[4]+1e-6f)>=3;
        }
        if(branches<3||junctions<1||!trunk||!horizontal) return 1;
    }
    return 0;
}
