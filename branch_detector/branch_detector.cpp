// Dependency-free C++17 core shared by native and wasm32 builds.
// DA2 values are inverse-like relative depth: larger means nearer. The
// reconstructed coordinates below are dimensionless pseudo-3D, never meters.
#include <stdint.h>

namespace {
constexpr int W = 320, H = 320, N = W * H, FULL = 640 * 640;
constexpr int MAX_BRANCHES = 254, FIELDS = 24, MAX_JUNCTIONS = 1024;
uint8_t rgb[FULL * 4], mask[FULL], labels[FULL];
float depth[FULL], rotation[9], branchData[MAX_BRANCHES * FIELDS];
int16_t junctionData[MAX_JUNCTIONS * 2];
uint8_t foreground[N], skeleton[N], removeMark[N], smallLabel[N];
uint8_t rawSkeleton[N];
float smallDepth[N];
uint32_t queue[N];
int branchCount, junctionCount;
float depthLow, depthHigh;

inline int minI(int a, int b) { return a < b ? a : b; }
inline int maxI(int a, int b) { return a > b ? a : b; }
inline float absF(float a) { return a < 0 ? -a : a; }
inline float clampF(float a, float lo, float hi) { return a < lo ? lo : a > hi ? hi : a; }
inline float sqrtF(float a) { return __builtin_sqrtf(a); }
inline int index(int x, int y) { return y * W + x; }

// Clockwise neighbours around a skeleton pixel, starting at north.
inline void neighbours(int x, int y, uint8_t p[8]) {
    p[0] = skeleton[index(x, y-1)]; p[1] = skeleton[index(x+1,y-1)];
    p[2] = skeleton[index(x+1,y)]; p[3] = skeleton[index(x+1,y+1)];
    p[4] = skeleton[index(x,y+1)]; p[5] = skeleton[index(x-1,y+1)];
    p[6] = skeleton[index(x-1,y)]; p[7] = skeleton[index(x-1,y-1)];
}
inline int degree(int x, int y) {
    uint8_t p[8]; neighbours(x,y,p);
    int transitions=0; for (int k=0;k<8;k++) transitions+=!p[k]&&p[(k+1)&7];
    return transitions;
}
// The inspection skeleton is derived from the segmentation mask alone. Its
// boundary neighbours are treated as background so edge pixels are retained.
inline void rawNeighbours(int x, int y, uint8_t p[8]) {
    const int dx[8]={0,1,1,1,0,-1,-1,-1};
    const int dy[8]={-1,-1,0,1,1,1,0,-1};
    for(int k=0;k<8;k++) {
        int nx=x+dx[k],ny=y+dy[k];
        p[k]=(nx>=0&&nx<W&&ny>=0&&ny<H)?rawSkeleton[index(nx,ny)]:0;
    }
}
inline void point(float x, float y, float d, float fx, float fy, float cx, float cy, float out[3]) {
    const float nearValue = clampF((d-depthLow)/(depthHigh-depthLow+1e-6f),0,1);
    const float z = 1.0f/(0.2f+0.8f*nearValue);
    const float px = ((x-cx)/fx)*z, py = -((y-cy)/fy)*z, pz = -z;
    for(int k=0;k<3;k++) out[k]=rotation[k*3]*px+rotation[k*3+1]*py+rotation[k*3+2]*pz;
}
}

extern "C" {
uint8_t* bd_rgb() { return rgb; }
uint8_t* bd_mask() { return mask; }
float* bd_depth() { return depth; }
float* bd_rotation() { return rotation; }
uint8_t* bd_labels() { return labels; }
uint8_t* bd_raw_skeleton() { return rawSkeleton; }
float* bd_branches() { return branchData; }
int16_t* bd_junctions() { return junctionData; }
int bd_branch_count() { return branchCount; }
int bd_junction_count() { return junctionCount; }
int bd_fields() { return FIELDS; }

// Unfiltered Zhang-Suen thinning of the 2x2-downsampled visible segmentation.
// This deliberately does not inspect depth, RGB, component size, or branch IDs.
int bd_run_raw_skeleton() {
    int xmin=W,xmax=-1,ymin=H,ymax=-1;
    for(int y=0;y<H;y++) for(int x=0;x<W;x++) {
        const int i=index(x,y),j=(y*2)*640+x*2;
        rawSkeleton[i]=(mask[j]||mask[j+1]||mask[j+640]||mask[j+641])?1:0;
        if(rawSkeleton[i]) { xmin=minI(xmin,x);xmax=maxI(xmax,x);ymin=minI(ymin,y);ymax=maxI(ymax,y); }
    }
    if(xmax<0) return 0;
    // A 320x320 shape can shrink by at most 160 layers; stop at convergence.
    for(int iter=0;iter<160;iter++) {
        int removed=0;
        for(int pass=0;pass<2;pass++) {
            for(int y=ymin;y<=ymax;y++) for(int x=xmin;x<=xmax;x++) {
                const int i=index(x,y);removeMark[i]=0;
                if(!rawSkeleton[i])continue;
                uint8_t p[8];rawNeighbours(x,y,p);int n=0,t=0;
                for(int k=0;k<8;k++){n+=p[k];t+=!p[k]&&p[(k+1)&7];}
                if(n<2||n>6||t!=1)continue;
                if(!pass) {if((p[0]&&p[2]&&p[4])||(p[2]&&p[4]&&p[6]))continue;}
                else {if((p[0]&&p[2]&&p[6])||(p[0]&&p[4]&&p[6]))continue;}
                removeMark[i]=1;
            }
            for(int y=ymin;y<=ymax;y++) for(int x=xmin;x<=xmax;x++) {
                const int i=index(x,y);
                if(removeMark[i]){rawSkeleton[i]=0;removed++;}
            }
        }
        if(!removed)break;
    }
    int count=0;
    for(int i=0;i<N;i++)count+=rawSkeleton[i]!=0;
    return count;
}

// Fields: id, score, length_px, area_px, width_px, straightness,
// inclination, depth_consistency, center_x/y, end0_x/y, end1_x/y,
// center_pseudo_xyz, mean_rgb, vertical_alignment, orientation_confidence,
// is_trunk, projected_up_alignment.
int bd_run(float fx, float fy, float cx, float cy, int worldUp) {
    branchCount=0; junctionCount=0;
    for(int i=0;i<N;i++) { foreground[i]=0; skeleton[i]=0; smallLabel[i]=0; }
    for(int i=0;i<FULL;i++) labels[i]=0;
    int hist[256]={}; int count=0, xmin=W, xmax=0, ymin=H, ymax=0;
    for(int y=1;y<H-1;y++) for(int x=1;x<W-1;x++) {
        int i=index(x,y), j=(y*2)*640+x*2;
        int covered=mask[j]||mask[j+1]||mask[j+640]||mask[j+641];
        if(!covered) continue;
        float d=0; int dn=0;
        const int offsets[4]={0,1,640,641};
        for(int t=0;t<4;t++) { float v=depth[j+offsets[t]]; if(v==v && v>-1e20f && v<1e20f) { d+=v;dn++; } }
        if(!dn) continue;
        foreground[i]=skeleton[i]=1; smallDepth[i]=d/dn;
        xmin=minI(xmin,x);xmax=maxI(xmax,x);ymin=minI(ymin,y);ymax=maxI(ymax,y);count++;
    }
    if(count<6) return 0;
    // Robust range from a fixed histogram after a linear min/max pass.
    float minD=1e30f,maxD=-1e30f;
    for(int i=0;i<N;i++) if(foreground[i]) { minD=smallDepth[i]<minD?smallDepth[i]:minD; maxD=smallDepth[i]>maxD?smallDepth[i]:maxD; }
    if(maxD-minD<1e-6f) maxD=minD+1.0f;
    for(int i=0;i<N;i++) if(foreground[i]) {
        int bin=(int)(255*(smallDepth[i]-minD)/(maxD-minD));hist[minI(255,maxI(0,bin))]++;
    }
    int seen=0; depthLow=minD; depthHigh=maxD;
    for(int b=0;b<256;b++) { seen+=hist[b]; if(seen>=count/50) {depthLow=minD+(maxD-minD)*b/255;break;} }
    seen=0;for(int b=255;b>=0;b--) {seen+=hist[b];if(seen>=count/50){depthHigh=minD+(maxD-minD)*b/255;break;} }
    if(depthHigh-depthLow<1e-6f) depthHigh=depthLow+1.0f;
    // Remove isolated specks before thinning. Eight-connectivity retains thin diagonals.
    for(int y=ymin;y<=ymax;y++) for(int x=xmin;x<=xmax;x++) {
        int seed=index(x,y);if(!foreground[seed]||smallLabel[seed])continue;
        int head=0,tail=0;queue[tail++]=seed;smallLabel[seed]=1;
        while(head<tail) {int p=queue[head++],px=p%W,py=p/W;
            for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
                if(!dx&&!dy)continue;
                int q=index(px+dx,py+dy);
                if(foreground[q]&&!smallLabel[q]){smallLabel[q]=1;queue[tail++]=q;}
            }
        }
        if(tail<8)for(int t=0;t<tail;t++) foreground[queue[t]]=skeleton[queue[t]]=0;
    }
    // Zhang-Suen two-pass thinning, bounded for predictable latency.
    for(int iter=0;iter<40;iter++) {
        int removed=0;
        for(int pass=0;pass<2;pass++) {
            for(int y=maxI(1,ymin);y<=minI(H-2,ymax);y++)for(int x=maxI(1,xmin);x<=minI(W-2,xmax);x++){
                int i=index(x,y);removeMark[i]=0;if(!skeleton[i])continue;
                uint8_t p[8];neighbours(x,y,p);int n=0,t=0;
                for(int k=0;k<8;k++){n+=p[k];t+=!p[k]&&p[(k+1)&7];}
                if(n<2||n>6||t!=1)continue;
                if(!pass) {if((p[0]&&p[2]&&p[4])||(p[2]&&p[4]&&p[6]))continue;}
                else {if((p[0]&&p[2]&&p[6])||(p[0]&&p[4]&&p[6]))continue;}
                removeMark[i]=1;
            }
            for(int y=maxI(1,ymin);y<=minI(H-2,ymax);y++)for(int x=maxI(1,xmin);x<=minI(W-2,xmax);x++){
                int i=index(x,y);if(removeMark[i]){skeleton[i]=0;removed++;}
            }
        }
        if(!removed)break;
    }
    // Mark junction cores. Branches are paths between cores or endpoints.
    for(int i=0;i<N;i++)removeMark[i]=0;
    for(int y=ymin;y<=ymax;y++)for(int x=xmin;x<=xmax;x++){
        int i=index(x,y);if(skeleton[i]&&degree(x,y)>=3)removeMark[i]=1;
    }
    for(int y=ymin;y<=ymax;y++)for(int x=xmin;x<=xmax;x++){
        int i=index(x,y);smallLabel[i]=0;if(!skeleton[i])continue;
        for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
            if(removeMark[index(x+dx,y+dy)])smallLabel[i]=255;
        }
    }
    for(int y=ymin;y<=ymax;y++)for(int x=xmin;x<=xmax;x++){
        int seed=index(x,y);if(smallLabel[seed]!=255)continue;
        int head=0,tail=0,sx=0,sy=0;queue[tail++]=seed;smallLabel[seed]=254;
        while(head<tail){int p=queue[head++],px=p%W,py=p/W;sx+=px;sy+=py;
            for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
                int q=index(px+dx,py+dy);if(smallLabel[q]==255){smallLabel[q]=254;queue[tail++]=q;}
            }
        }
        if(junctionCount<MAX_JUNCTIONS){junctionData[junctionCount*2]=(int16_t)(sx*2/tail);junctionData[junctionCount*2+1]=(int16_t)(sy*2/tail);junctionCount++;}
    }
    // Trace each connected skeleton path. Width/depth/color statistics are
    // calculated on the assigned foreground after geodesic expansion below.
    for(int y=ymin;y<=ymax;y++)for(int x=xmin;x<=xmax;x++){
        int seed=index(x,y);if(!skeleton[seed]||smallLabel[seed])continue;
        int head=0,tail=0,ex0=-1,ex1=-1,sx=0,sy=0;
        queue[tail++]=seed;smallLabel[seed]=1;
        while(head<tail){int p=queue[head++],px=p%W,py=p/W;sx+=px;sy+=py;
            int pathDegree=0;
            for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
                if(!dx&&!dy)continue;
                int q=index(px+dx,py+dy);
                if(skeleton[q]&&smallLabel[q]!=254){pathDegree++;if(!smallLabel[q]){smallLabel[q]=1;queue[tail++]=q;}}
            }
            if(pathDegree<=1){if(ex0<0)ex0=p;else ex1=p;}
        }
        if(tail<6||branchCount>=253){for(int t=0;t<tail;t++)smallLabel[queue[t]]=254;continue;}
        if(ex0<0)ex0=seed;
        if(ex1<0)ex1=queue[tail-1];
        int id=++branchCount;float* b=branchData+(id-1)*FIELDS;
        for(int k=0;k<FIELDS;k++)b[k]=0;
        b[0]=(float)id;b[2]=(float)(tail*2);b[8]=(float)(sx*2/tail);b[9]=(float)(sy*2/tail);
        b[10]=(float)((ex0%W)*2);b[11]=(float)((ex0/W)*2);
        b[12]=(float)((ex1%W)*2);b[13]=(float)((ex1/W)*2);
        float ddx=b[12]-b[10],ddy=b[13]-b[11];
        b[5]=clampF(sqrtF(ddx*ddx+ddy*ddy)/(b[2]+1e-6f),0,1);
        float a[3],c[3];point(b[10],b[11],smallDepth[ex0],fx,fy,cx,cy,a);
        point(b[12],b[13],smallDepth[ex1],fx,fy,cx,cy,c);
        float vx=c[0]-a[0],vy=c[1]-a[1],vz=c[2]-a[2];
        float magnitude=sqrtF(vx*vx+vy*vy+vz*vz)+1e-6f;
        b[6]=absF(worldUp==2?vz:vy)/magnitude;
        for(int t=0;t<tail;t++)smallLabel[queue[t]]=(uint8_t)id;
    }
    // Multi-source geodesic fill assigns each foreground pixel to the nearest
    // centerline path without crossing background. Junctions may be assigned.
    int head=0,tail=0;
    for(int i=0;i<N;i++)if(smallLabel[i]>0&&smallLabel[i]<254&&skeleton[i])queue[tail++]=i;
    while(head<tail){int p=queue[head++];
        const int offset[4]={-W,-1,1,W};
        for(int k=0;k<4;k++){int q=p+offset[k];if(q<0||q>=N||!foreground[q]||(smallLabel[q]!=0&&smallLabel[q]!=254))continue;
            // Keep large relative-depth discontinuities from leaking across crossings.
            if(absF(smallDepth[p]-smallDepth[q])>0.30f*(depthHigh-depthLow))continue;
            smallLabel[q]=smallLabel[p];queue[tail++]=q;
        }
    }
    int area[MAX_BRANCHES]={};float sumD[MAX_BRANCHES]={},sumD2[MAX_BRANCHES]={};
    float pseudoSum[MAX_BRANCHES*3]={};
    float color[MAX_BRANCHES*3]={};
    int lineCount[MAX_BRANCHES]={};
    float lineSum[MAX_BRANCHES*3]={},lineCov[MAX_BRANCHES*6]={};
    float imageSum[MAX_BRANCHES*2]={},imageCov[MAX_BRANCHES*3]={};
    for(int y=0;y<H;y++)for(int x=0;x<W;x++){
        int i=index(x,y),id=smallLabel[i];if(id<1||id>branchCount)continue;
        int p=(y*2)*640+x*2;float d=smallDepth[i];
        area[id-1]++;sumD[id-1]+=d;sumD2[id-1]+=d*d;
        float xyz[3];point(x*2+1,y*2+1,d,fx,fy,cx,cy,xyz);
        for(int c=0;c<3;c++)pseudoSum[(id-1)*3+c]+=xyz[c];
        if(skeleton[i]) {
            int b=id-1;lineCount[b]++;
            for(int c=0;c<3;c++)lineSum[b*3+c]+=xyz[c];
            float* cov=lineCov+b*6;
            cov[0]+=xyz[0]*xyz[0];cov[1]+=xyz[0]*xyz[1];cov[2]+=xyz[0]*xyz[2];
            cov[3]+=xyz[1]*xyz[1];cov[4]+=xyz[1]*xyz[2];cov[5]+=xyz[2]*xyz[2];
            float px=x*2+1,py=y*2+1;float* ic=imageCov+b*3;
            imageSum[b*2]+=px;imageSum[b*2+1]+=py;
            ic[0]+=px*px;ic[1]+=px*py;ic[2]+=py*py;
        }
        for(int c=0;c<3;c++)color[(id-1)*3+c]+=rgb[p*4+c];
        for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++){
            int q=p+dy*640+dx;if(mask[q])labels[q]=(uint8_t)id;
        }
    }
    for(int id=1;id<=branchCount;id++){
        float* b=branchData+(id-1)*FIELDS;int n=area[id-1];b[3]=(float)(n*4);
        b[4]=b[3]/(b[2]+1e-6f);float mean=n?sumD[id-1]/n:0;
        float variance=n?sumD2[id-1]/n-mean*mean:0;
        b[7]=clampF(1-sqrtF(variance>0?variance:0)/(depthHigh-depthLow+1e-6f),0,1);
        for(int c=0;c<3;c++)b[14+c]=n?pseudoSum[(id-1)*3+c]/n:0;
        for(int c=0;c<3;c++)b[17+c]=n?color[(id-1)*3+c]/n:0;
    }
    // Fit each full centerline, not just its two endpoints. Relative-depth
    // distortion can dominate its 3D PCA; use the median-depth image tangent
    // when the pseudo-3D line is inconsistent.
    const int up=worldUp==2?2:1;
    float ux=fx*rotation[up*3],uy=-fy*rotation[up*3+1];
    float un=sqrtF(ux*ux+uy*uy);
    if(un<1e-5f){ux=0;uy=-1;}else{ux/=un;uy/=un;}
    float low=1e30f,high=-1e30f;
    for(int k=0;k<4;k++){
        float xx=(k&1)?xmax*2:xmin*2,yy=(k&2)?ymax*2:ymin*2;
        float p=xx*ux+yy*uy;if(p<low)low=p;if(p>high)high=p;
    }
    float widths[MAX_BRANCHES];int widthCount=0;
    for(int id=1;id<=branchCount;id++)if(branchData[(id-1)*FIELDS+2]>=16)widths[widthCount++]=branchData[(id-1)*FIELDS+4];
    for(int i=1;i<widthCount;i++){float v=widths[i];int j=i-1;while(j>=0&&widths[j]>v){widths[j+1]=widths[j];j--;}widths[j+1]=v;}
    const float medianWidth=widthCount?widths[widthCount/2]:8;
    for(int id=1;id<=branchCount;id++){
        int a=id-1,n=lineCount[a];float* b=branchData+a*FIELDS;
        if(n<3)continue;
        float mx=imageSum[a*2]/n,my=imageSum[a*2+1]/n;
        float xx=imageCov[a*3]/n-mx*mx,xy=imageCov[a*3+1]/n-mx*my,yy=imageCov[a*3+2]/n-my*my;
        float vx=xx>=yy?1:0,vy=xx>=yy?0:1;
        for(int t=0;t<8;t++){float nx=xx*vx+xy*vy,ny=xy*vx+yy*vy,norm=sqrtF(nx*nx+ny*ny);
            if(norm<1e-8f)break;
            vx=nx/norm;vy=ny/norm;}
        b[23]=clampF(absF(vx*ux+vy*uy),0,1);
        float imageEigen=vx*(xx*vx+xy*vy)+vy*(xy*vx+yy*vy);
        float imageConfidence=clampF((2*imageEigen/(xx+yy+1e-6f)-1),0,1);
        float mean[3];for(int c=0;c<3;c++)mean[c]=lineSum[a*3+c]/n;
        float* s=lineCov+a*6;
        float cxx=s[0]/n-mean[0]*mean[0],cxy=s[1]/n-mean[0]*mean[1],cxz=s[2]/n-mean[0]*mean[2];
        float cyy=s[3]/n-mean[1]*mean[1],cyz=s[4]/n-mean[1]*mean[2],czz=s[5]/n-mean[2]*mean[2];
        float axis[3]={cxx>=cyy&&cxx>=czz?1.0f:0.0f,cyy>cxx&&cyy>=czz?1.0f:0.0f,czz>cxx&&czz>cyy?1.0f:0.0f};
        for(int t=0;t<10;t++){
            float next[3]={cxx*axis[0]+cxy*axis[1]+cxz*axis[2],cxy*axis[0]+cyy*axis[1]+cyz*axis[2],cxz*axis[0]+cyz*axis[1]+czz*axis[2]};
            float norm=sqrtF(next[0]*next[0]+next[1]*next[1]+next[2]*next[2]);
            if(norm<1e-8f)break;
            for(int c=0;c<3;c++)axis[c]=next[c]/norm;
        }
        float eigen=axis[0]*(cxx*axis[0]+cxy*axis[1]+cxz*axis[2])+axis[1]*(cxy*axis[0]+cyy*axis[1]+cyz*axis[2])+axis[2]*(cxz*axis[0]+cyz*axis[1]+czz*axis[2]);
        float pseudoConfidence=clampF((1.5f*eigen/(cxx+cyy+czz+1e-6f)-0.5f),0,1);
        float ray[3];
        for(int c=0;c<3;c++)ray[c]=rotation[c*3]*(vx/fx)+rotation[c*3+1]*(-vy/fy);
        float rayNorm=sqrtF(ray[0]*ray[0]+ray[1]*ray[1]+ray[2]*ray[2])+1e-6f;
        float rayVertical=absF(ray[up]/rayNorm),pseudoVertical=absF(axis[up]);
        // Monocular relative depth can bend an otherwise straight centerline.
        // Prefer the camera-corrected image tangent when the two disagree.
        float vertical=(pseudoConfidence>=0.7f&&b[7]>=0.85f&&absF(pseudoVertical-rayVertical)<=0.25f)?pseudoVertical:rayVertical;
        b[6]=b[20]=clampF(vertical,0,1);
        b[21]=clampF(imageConfidence*(pseudoConfidence>=0.7f?pseudoConfidence:0.75f),0,1);
        float lengthScore=clampF(b[2]/180,0,1),widthScore=clampF(b[4]/12,0,1);
        float slenderness=clampF((b[2]/(b[4]+1e-6f)-1)/4,0,1);
        b[1]=(0.35f*lengthScore+0.2f*b[5]+0.2f*(1-b[6])+0.15f*b[7]+0.1f*widthScore)*slenderness;
        float projectedCenter=b[8]*ux+b[9]*uy;
        bool lower=(projectedCenter-low)/(high-low+1e-6f)<0.68f;
        bool wide=b[4]>=maxI(12,(int)(medianWidth*1.6f));
        b[22]=(b[2]>=12&&wide&&lower&&(b[20]>=0.7f||b[23]>=0.78f))?1:0;
    }
    // Continue a broad upright trunk through nearby junction paths. Do not
    // follow narrow lateral limbs.
    for(int iter=0;iter<branchCount;iter++){
        int changed=0;
        for(int j=0;j<junctionCount;j++){
            bool nearby[MAX_BRANCHES]={};int cx=junctionData[j*2]/2,cy=junctionData[j*2+1]/2;
            for(int dy=-3;dy<=3;dy++)for(int dx=-3;dx<=3;dx++){
                int x=cx+dx,y=cy+dy;if(x<0||x>=W||y<0||y>=H)continue;
                int p=index(x,y),id=smallLabel[p];if(skeleton[p]&&id>0&&id<=branchCount)nearby[id-1]=true;
            }
            float trunkWidth=0;
            for(int id=0;id<branchCount;id++)if(nearby[id]&&branchData[id*FIELDS+22]>0&&branchData[id*FIELDS+4]>trunkWidth)trunkWidth=branchData[id*FIELDS+4];
            if(!trunkWidth)continue;
            for(int id=0;id<branchCount;id++)if(nearby[id]&&!branchData[id*FIELDS+22]){
                float* b=branchData+id*FIELDS;
                if(b[2]>=12&&b[4]>=maxI(8,(int)(trunkWidth*0.5f))&&(b[20]>=0.62f||b[23]>=0.7f)) {b[22]=1;changed++;}
            }
        }
        if(!changed)break;
    }
    return branchCount;
}
}
