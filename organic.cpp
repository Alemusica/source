// organic.noise~ — Max/MSP external built with Cycling '74 Min-API (C++)
// v0.3.3 "AVANT-FIX++"
// Micro-fix: IDFT indexing con Nloc, clamp q in OT, undenorm in pull_ola, assert dev su ring OLA.
// Mantiene i fix precedenti: YIN (τ→bin)+median(3), STFT/OLA hop fisso, no alloc in callback,
// sqrt-Hann (COLA), twiddle precomputati, Poisson clamp, limiter per-istanza fast-attack, denorm guards.

#include "c74_min.h"
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cassert>   // per assert di sviluppo (disabilitato in release con NDEBUG)

using namespace c74::min;

static constexpr double PI  = 3.14159265358979323846;
static constexpr double PHI = 1.6180339887498948482;

//--------------------- Utility RNG ---------------------//
struct XorShift32 {
    uint32_t state { 0x12345678u };
    void seed(uint32_t s) { state = s ? s : 0xdeadbeefu; }
    inline uint32_t nextu() {
        uint32_t x = state;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5; state = x; return x;
    }
    inline double next() { return (nextu() >> 8) * (1.0/16777216.0); }
    inline double bipolar() { return next()*2.0 - 1.0; }
};

//--------------------- Small helpers ---------------------//
inline static double clamp01(double v){ return v<0.0?0.0:(v>1.0?1.0:v); }
inline static double mix_lin(double a, double b, double t){ return a + (b-a)*t; }
inline static double softsat(double x){ return std::tanh(x); }
inline static double sgn(double x){ return (x<0.0)?-1.0:1.0; }
// Denormals guard (RT safety)
inline static double undenorm(double x){ return (std::fabs(x) < 1e-30) ? 0.0 : x; }

//--------------------- Simple filters ---------------------//
struct OnePole { double a{0.0}, b{1.0}, z{0.0};
    void set_lp(double cutoff, double sr) { cutoff = std::max(0.1, cutoff); double p = std::exp(-2.0 * PI * cutoff / sr); a = 1.0 - p; b = p; }
    inline double process(double x){ z = a*x + b*z; return z; }
};

struct DCBlock { double R{0.995}, x1{0.0}, y1{0.0};
    inline double process(double x){ double y = x - x1 + R*y1; x1=x; y1=y; return undenorm(y); }
};

struct ParamSmoother {
    double z{0.0};
    double a{0.0};
    void setup(double ms, double sr) {
        double tau = std::max(1.0, (ms / 1000.0) * sr);
        a = std::exp(-1.0 / tau);
        z = 0.0;
    }
    double process(double target) {
        z = a * z + (1.0 - a) * target;
        return z;
    }
};

//--------------------- Pink noise (Voss-McCartney) ---------------------//
struct PinkVoss {
    static constexpr int N=5;
    std::array<double,N> rows{}; uint32_t counter{0}; XorShift32 rng;
    void reseed(uint32_t s){ rng.seed(s? s : 0x87654321u); counter=0; rows.fill(0.0); }
    inline int ctz(uint32_t v){ int n=0; while(((v>>n)&1u)==0u && n<31) ++n; return n; }
    inline double process(){
        uint32_t c = ++counter;
        int n_zeros = (c==0) ? (N-1) : ctz(c);
        if(n_zeros>=N) n_zeros=N-1;
        for(int i=0;i<=n_zeros;i++) rows[i]=rng.bipolar();
        double s=0.0; for(double r:rows) s+=r; return s/double(N);
    }
};

//--------------------- Brown noise (leaky integrator) ---------------------//
struct Brown {
    double y{0.0}; DCBlock dc;
    inline double process(double white){
        y += white*0.02; if(y>1.0) y=1.0; else if(y<-1.0) y=-1.0;
        return dc.process(y);
    }
};

//--------------------- Ornstein-Uhlenbeck ---------------------//
struct OU {
    double x{0.0}; double theta{1.5}; double sigma{0.2}; double sr{48000.0}; XorShift32 rng;
    void reseed(uint32_t s){ rng.seed(s? s : 0xCAFEBABEu); }
    inline void set(double thz, double sig, double srr){ theta=std::max(0.0001,thz); sigma=sig; sr=srr; }
    inline double process(){
        double dt=1.0/sr; double u1=std::max(1e-12,rng.next()); double u2=rng.next();
        double g=std::sqrt(-2.0*std::log(u1))*std::cos(2*PI*u2);
        x += theta*(0.0-x)*dt + sigma*std::sqrt(dt)*g;
        if(x>1.5)x=1.5; if(x<-1.5)x=-1.5; return std::tanh(x);
    }
};

//--------------------- Envelope follower ---------------------//
struct EnvFollow {
    double env{0.0}; double a_a{0.0}, a_r{0.0};
    void setup(double attack_ms, double release_ms, double sr){
        auto calc=[sr](double ms){ return std::exp(-1.0/(((ms/1000.0)*sr)+1e-9)); };
        a_a=calc(std::max(0.01,attack_ms)); a_r=calc(std::max(0.01,release_ms));
    }
    inline double process(double x){
        double e_in=std::fabs(x); double c=(e_in>env)?a_a:a_r; env=env*c + e_in*(1.0-c); return env;
    }
};

//--------------------- Logistic map (chaos) ---------------------//
struct Logistic { double x{0.1234}; double r{3.7}; inline void set_r(double rr){ r=rr; } inline double step(){ x=r*x*(1.0-x); return x; } };

//===================== Fractional color noise (1/f^β) =====================//
// y[n] ≈ (1 - z^{-1})^{-d} * white, d = β/2 (troncato a K)
struct FractionalColorNoise {
    int K{64}; double beta{1.0}; double d{0.5};
    std::vector<double> c; std::vector<double> ring; int wr{0};
    XorShift32 rng;
    void setup(int orderK, double beta_, int ringlen=8192, uint32_t seed=0xA1B2C3D4u){
        K = std::max(4, orderK); beta = std::max(0.0, std::min(2.5, beta_)); d = 0.5*beta;
        c.assign(K+1,0.0); ring.assign(ringlen,0.0); wr=0; rng.seed(seed);
        c[0]=1.0; for(int k=1;k<=K;k++){ c[k] = c[k-1] * (d + (k-1)) / double(k); }
    }
    inline double process(){
        ring[wr] = rng.bipolar();
        double y = 0.0; int N = (int)ring.size(); int idx = wr;
        for(int k=0;k<=K;k++){ y += c[k] * ring[idx]; if(--idx<0) idx += N; }
        if(++wr >= (int)ring.size()) wr = 0;
        return std::tanh(0.35*y);
    }
};

//===================== Tiny DFT Engine (sqrt-Hann + twiddle) =====================//
struct TinyDFT {
    int N{128}; int H{64};                   // frame & hop (H=N/2)
    std::vector<double> win;                 // sqrt-Hann (COLA con H=N/2)
    std::vector<double> coskn, sinkn;        // twiddle pretabellati: size N*N

    TinyDFT(int n=128, int h=64):N(n),H(h){
        win.resize(N);
        for(int n_=0;n_<N;n_++){
            double w = 0.5 - 0.5*std::cos(2.0*PI*(double)n_/N);
            win[n_] = std::sqrt(w + 1e-12); // sqrt-Hann
        }
        // precompute twiddles
        coskn.resize(N*N); sinkn.resize(N*N);
        for(int k=0;k<N;k++){
            for(int n_=0;n_<N;n_++){
                double ang = 2.0*PI*(double)k*(double)n_/(double)N;
                coskn[k*N + n_] = std::cos(ang);
                sinkn[k*N + n_] = std::sin(ang);
            }
        }
    }

    inline void dft(const std::vector<double>& x, std::vector<double>& re, std::vector<double>& im){
        for(int k=0;k<N;k++){
            double rk=0.0, ik=0.0;
            const double* cptr = &coskn[k*N];
            const double* sptr = &sinkn[k*N];
            for(int n_=0;n_<N;n_){
                double xn = x[n_] * win[n_];
                rk += xn *  cptr[n_];    // cos(+)
                ik -= xn *  sptr[n_];    // sin(-) per forward
            }
            re[k]=rk; im[k]=ik;
        }
    }
    inline void idft(const std::vector<double>& re, const std::vector<double>& im, std::vector<double>& y){
        const int Nloc = (int)win.size();
        for(int n_=0;n_<Nloc;n_){
            double s=0.0;
            for(int k=0;k<Nloc;k++){
                // usa Nloc per coerenza future-proof con eventuali refactor
                double c  = coskn[k*Nloc + n_];
                double s_ = sinkn[k*Nloc + n_];
                s += re[k]*c - im[k]*s_;
            }
            // sqrt-Hann in sintesi, norma 1/N
            y[n_] = (s*(1.0/Nloc)) * win[n_];
        }
    }
    void copy_long_window(const std::vector<double>& ring, int wr, std::vector<double>& dst, int stride=1) {
        const int Nlong = (int)dst.size();
        int idx = wr - Nlong;
        if(idx < 0) idx += (int)ring.size();
        for(int n=0; n<Nlong; n+=stride) {
            dst[n] = ring[(idx+n) % (int)ring.size()];
        }
    }
};

//===================== YIN f0 estimator (prealloc, ritorna τ) =====================//
struct Yin {
    double thresh{0.10};
    int tau_min{2}, tau_max{128};
    std::vector<double> d, cmnd;
    void prepare(int N){
        tau_max = std::min(128, N/2);
        d.assign(tau_max+1, 0.0);
        cmnd.assign(tau_max+1, 0.0);
    }
    // Restituisce τ (lag in campioni)
    double estimate_tau(const std::vector<double>& x){
        const int N=(int)x.size();
        const int max_tau = std::min(tau_max, N/2);
        if(max_tau<=tau_min+2) return 2.0;

        std::fill(d.begin(), d.end(), 0.0);
        for(int tau=1; tau<=max_tau; ++tau){
            double sum=0.0;
            for(int n=0; n<N-tau; ++n){
                double diff = x[n]-x[n+tau];
                sum += diff*diff;
            }
            d[tau]=sum;
        }
        double cum=0.0; cmnd[0]=1.0;
        for(int tau=1; tau<=max_tau; ++tau){
            cum += d[tau];
            cmnd[tau] = d[tau] * tau / std::max(1e-12, cum);
        }
        int best = tau_min;
        for(int tau=tau_min+1; tau<=max_tau; ++tau){
            if(cmnd[tau] < thresh){
                int t0 = std::max(tau-1, tau_min);
                int t2 = std::min(tau+1, max_tau);
                double a = cmnd[t0], b=cmnd[tau], c=cmnd[t2];
                double denom = (a - 2*b + c);
                double offset = (std::fabs(denom)>1e-12) ? 0.5*(a - c)/denom : 0.0;
                best = std::max(tau_min, std::min(max_tau, (int)std::round(tau + offset)));
                break;
            }
        }
        return (double)best;
    }
};

//===================== Optimal Transport 1D (prealloc) =====================//
struct OT1D {
    std::vector<double> cfx, cfg;
    void prepare(int N){
        cfx.assign(N,0.0); cfg.assign(N,0.0);
    }
    inline static int inv_cdf(const std::vector<double>& cdf, double q){
        int lo=0, hi=(int)cdf.size()-1, ans=hi;
        while(lo<=hi){ int mid=(lo+hi)/2; if(cdf[mid] >= q){ ans=mid; hi=mid-1; } else lo=mid+1; }
        return ans;
    }
    // out = barycenter(X,G; tau) — tutte size N, normalizza internamente
    void barycenter_push(const std::vector<double>& Xraw, const std::vector<double>& Graw, std::vector<double>& out, double tau){
        const int N=(int)out.size();
        // copia e normalizza non negativa
        double sx=0.0, sg=0.0;
        for(int k=0;k<N;k++){ double vx=std::max(0.0,Xraw[k]); cfx[k]=vx; sx+=vx; }
        for(int k=0;k<N;k++){ double vg=std::max(0.0,Graw[k]); cfg[k]=vg; sg+=vg; }
        sx = (sx<=1e-12)?1.0:sx; sg = (sg<=1e-12)?1.0:sg;
        for(int k=0;k<N;k++){ cfx[k]/=sx; cfg[k]/=sg; }
        // CDF
        double acc=0.0; for(int k=0;k<N;k++){ acc+=cfx[k]; cfx[k]=acc; }
        acc=0.0; for(int k=0;k<N;k++){ acc+=cfg[k]; cfg[k]=acc; }

        // push-forward con massa del quantile di X
        std::fill(out.begin(), out.end(), 0.0);
        for(int k=0;k<N;k++){
            // clamp numerico per robustezza
            double q = cfx[k];
            q = std::min(1.0 - 1e-12, std::max(0.0, q));
            int T  = inv_cdf(cfg, q);
            double t = (1.0 - tau)*k + tau*(double)T;
            int k0 = std::min(N-1, std::max(0, (int)std::floor(t)));
            int k1 = std::min(N-1, k0+1);
            double w1 = t - (double)k0; double w0 = 1.0 - w1;
            double mass = (k ? (cfx[k] - cfx[k-1]) : cfx[0]); // massa del k-esimo quantile
            out[k0] += w0 * mass;
            out[k1] += w1 * mass;
        }
        // rinormalizza
        double sp=0.0; for(double v:out) sp+=v; if(sp>1e-12) for(double &v:out) v/=sp;
    }
};

//===================== Non-homogeneous Poisson scheduler =====================//
struct NHPPoisson {
    double sr{48000.0}; XorShift32 rng;
    void setup(double srr, uint32_t seed){ sr=srr; rng.seed(seed? seed : 0xDEADBEEFu); }
    inline bool event(double lambda_inst){
        double lam = std::max(0.0, lambda_inst);
        // clamp robustezza: p = lam/sr ≤ 0.25
        lam = std::min(lam, 0.25*sr);
        return rng.next() < (lam / sr);
    }
};

//===================== micro-FDN 4x (Hadamard) =====================//
struct BiquadFilter {
    double b0{1.0},b1{0.0},b2{0.0},a1{0.0},a2{0.0};
    double x1{0.0},x2{0.0},y1{0.0},y2{0.0};
    void set_shelving(double fc, double gain_db, double sr) {
        double w0 = 2.0*PI*fc/sr;
        double A = std::pow(10.0, gain_db/40.0);
        double alpha = std::sin(w0)/2.0 * std::sqrt((A + 1.0/A)*(1.0/0.707 - 1.0) + 2.0);
        double two_sqrtA_alpha = 2.0*std::sqrt(A)*alpha;
        b0 = A*((A+1.0) + (A-1.0)*std::cos(w0) + two_sqrtA_alpha);
        b1 = -2.0*A*((A-1.0) + (A+1.0)*std::cos(w0));
        b2 = A*((A+1.0) + (A-1.0)*std::cos(w0) - two_sqrtA_alpha);
        a1 = 2.0*((A-1.0) - (A+1.0)*std::cos(w0));
        a2 = (A+1.0) + (A-1.0)*std::cos(w0) - two_sqrtA_alpha;
        double a0 = (A+1.0) - (A-1.0)*std::cos(w0) + two_sqrtA_alpha;
        b0 /= a0; b1 /= a0; b2 /= a0; a1 /= a0; a2 /= a0;
    }
    inline double process(double x) {
        double y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2;
        x2=x1; x1=x; y2=y1; y1=y;
        return y;
    }
};

struct FDN4 {
    std::vector<double> d0,d1,d2,d3;
    int w0{0},w1{0},w2{0},w3{0};
    double g{0.6}; double sr{48000.0};
    BiquadFilter shelf0, shelf1, shelf2, shelf3;
    void setup(double srr, double t0_ms=29.7, double t1_ms=37.1, double t2_ms=41.3, double t3_ms=43.9, double gfb=0.6){
        sr=srr; g = std::min(0.95, std::max(0.1, gfb));
        auto len=[&](double ms){ int L=(int)std::round(sr*(ms/1000.0)); return std::max(32,L); };
        d0.assign(len(t0_ms),0.0); d1.assign(len(t1_ms),0.0); d2.assign(len(t2_ms),0.0); d3.assign(len(t3_ms),0.0);
        w0=w1=w2=w3=0;
        // Initialize shelving filters
        shelf0.set_shelving(2000.0, -3.0, sr);
        shelf1.set_shelving(2200.0, -3.0, sr);
        shelf2.set_shelving(2400.0, -3.0, sr);
        shelf3.set_shelving(2600.0, -3.0, sr);
    }
    inline double tick(double x){
        double y0=shelf0.process(d0[w0]), y1=shelf1.process(d1[w1]);
        double y2=shelf2.process(d2[w2]), y3=shelf3.process(d3[w3]);
        double m0 = 0.5*( y0 + y1 + y2 + y3 );
        double m1 = 0.5*( y0 - y1 + y2 - y3 );
        double m2 = 0.5*( y0 + y1 - y2 - y3 );
        double m3 = 0.5*( y0 - y1 - y2 + y3 );
        d0[w0] = x + g*m0; if(++w0>=(int)d0.size()) w0=0;
        d1[w1] = x + g*m1; if(++w1>=(int)d1.size()) w1=0;
        d2[w2] = x + g*m2; if(++w2>=(int)d2.size()) w2=0;
        d3[w3] = x + g*m3; if(++w3>=(int)d3.size()) w3=0;
        return undenorm(0.25*(m0+m1+m2+m3));
    }
};

//===================== Main External =====================//
class organic_noise_tilde : public object<organic_noise_tilde>, public vector_operator<> {
public:
    MIN_DESCRIPTION { "Organic noise & spectral clouds (SMR) con HARMONIA, fGn/fBm, YIN (τ→bin), OT barycenter, FDN unitario." };
    MIN_TAGS { "noise, spectral, asmr, emdr, ot, yin, fdn, dsp" };
    MIN_AUTHOR { "you" };

    inlet<>  in1 { this, "signal in (mono)" };
    outlet<> out1 { this, "left out", "signal" };
    outlet<> out2 { this, "right out", "signal" };

    // --------- Global mode
    attribute<symbol> mode { this, "mode", "core", description { "core = time-domain; smr = spectral match & residue." } };

    // --------- Core
    attribute<number> mix { this, "mix", 0.5, description { "0=dry, 1=wet." } };
    attribute<number> color { this, "color", 0.5, range {0.0,1.0} };
    attribute<number> fbm_depth { this, "fbm_depth", 0.25, range {0.0,1.0} };
    attribute<number> ou_rate { this, "ou_rate", 2.0, range {0.01,20.0} };
    attribute<number> ou_sigma { this, "ou_sigma", 0.3, range {0.0,2.0} };
    attribute<number> chaos { this, "chaos", 0.2, range {0.0,1.0} };
    attribute<number> glitch_rate { this, "glitch_rate", 6.0, range {0.1,50.0} };
    attribute<number> glitch_depth { this, "glitch_depth", 0.6, range {0.0,1.0} };
    attribute<number> env_sense { this, "env_sense", 0.7, range {0.0,2.0} };
    attribute<number> emdr_rate { this, "emdr_rate", 1.0, range {0.1,8.0} };
    attribute<number> emdr_depth { this, "emdr_depth", 0.3, range {0.0,1.0} };
    attribute<number> seed { this, "seed", 1337 };

    // --------- SMR Tingles
    attribute<number> smr_alpha   { this, "smr_alpha", 0.55, range {0.0,1.0} };
    attribute<number> grain_width { this, "grain_width", 6.0, range {1.0, 32.0} };
    attribute<number> grain_rate  { this, "grain_rate", 40.0, range {5.0, 200.0} };
    attribute<number> harmonicity { this, "harmonicity", 0.0, range {0.0, 0.03} };
    attribute<number> gate_thresh { this, "gate_thresh", 0.15, range {0.0,1.0} };
    attribute<number> phi_dither  { this, "phi_dither", 0.0042, range {0.0,0.05} };

    // --------- HARMONIA (benessere)
    attribute<number> harmonia       { this, "harmonia", 1.0, range{0.0,1.0} };
    attribute<number> breathe_rate   { this, "breathe_rate", 0.10, range{0.02,0.33} };
    attribute<number> breathe_depth  { this, "breathe_depth", 0.6,  range{0.0,1.0} };
    attribute<number> crossfeed      { this, "crossfeed", 0.12, range{0.0,0.5} };
    attribute<number> cf_cutoff      { this, "cf_cutoff", 700.0, range{120.0,3000.0} };
    attribute<number> tp_ceiling     { this, "tp_ceiling", -1.0, range{-3.0,-0.1} };

    // --------- fGn/fBm (1/f^β)
    attribute<number> beta_1f        { this, "beta", 1.0, range{0.0,2.5} };
    attribute<number> beta_mix       { this, "beta_mix", 0.25, range{0.0,1.0} };

    // --------- Poisson/OT/YIN
    attribute<number> poisson_base   { this, "poisson_base", 20.0, range{0.0,300.0} };
    attribute<number> poisson_envamt { this, "poisson_envamt", 40.0, range{0.0,300.0} };
    attribute<number> ot_enable      { this, "ot_enable", 1.0, range{0.0,1.0} };
    attribute<number> ot_tau         { this, "ot_tau", 0.5, range{0.0,1.0} };
    attribute<number> yin_enable     { this, "yin_enable", 1.0, range{0.0,1.0} };
    attribute<number> yin_thresh     { this, "yin_thresh", 0.10, range{0.02,0.3} };

    // preset
    message<> preset { this, "preset", "", MIN_FUNCTION {
        if(args.size() && args[0] == symbol{"asmr_tingles"}){
            mode = symbol{"smr"};
            mix = 0.65; color = 0.35; fbm_depth = 0.22;
            smr_alpha = 0.6; grain_width = 7.0; grain_rate = 55.0; harmonicity = 0.004; gate_thresh = 0.12; phi_dither = 0.0038;
            emdr_rate = 1.0; emdr_depth = 0.3; env_sense = 0.9; glitch_depth = 0.35; glitch_rate = 8.0; chaos = 0.12;
            harmonia = 1.0; breathe_rate = 0.10; breathe_depth = 0.6; crossfeed = 0.12; tp_ceiling = -1.0;
            beta_1f = 1.0; beta_mix = 0.25;
            poisson_base = 20.0; poisson_envamt = 40.0; ot_enable = 1.0; ot_tau = 0.5; yin_enable = 1.0; yin_thresh = 0.10;
        }
        return {};
    } };

    // DSP setup
    message<> dspsetup { this, "dspsetup", MIN_FUNCTION {
        m_sr = args[0]; inv_sr = 1.0/m_sr;

        env.setup(4.0, 60.0, m_sr);
        for(int i=0;i<3;i++){ fbm_lps[i].set_lp((i==0?0.5:(i==1?2.0:8.0)), m_sr); }
        ou.set(ou_rate, ou_sigma, m_sr);
        // seed propagation
        rng.seed((uint32_t)seed);
        pink.reseed((uint32_t)seed ^ 0x13579BDFu);
        ou.reseed((uint32_t)seed ^ 0xCAFEBABEu);
        frac.setup(64, beta_1f, 8192, (uint32_t)seed ^ 0xA1B2C3D4u);
        lmap.set_r(3.5 + chaos*0.4);

        // STFT/OLA
        dft = TinyDFT(128, 64); // H=N/2
        hop_count = 0; need_reloc = true;
        active_config.prepare(m_sr);
        shadow_config.prepare(m_sr);

        inring.assign(4096, 0.0); in_wr=0;
        olaring.assign(4096, 0.0); ola_rd=0; ola_wr=0;

#ifndef NDEBUG
        // Dev-safety: ring OLA almeno 4 frame
        assert(olaring.size() >= (size_t)(4 * dft.N));
#endif

        active_config.yin.thresh = yin_thresh;
        shadow_config.yin.thresh = yin_thresh;
        active_config.yin.prepare(dft.N);
        shadow_config.yin.prepare(dft.N);

        active_config.ot.prepare(dft.N);
        shadow_config.ot.prepare(dft.N);

        // HARMONIA
        breathe_phase=0.0;
        tp_lin = std::pow(10.0, (double)tp_ceiling / 20.0);
        double rel_ms = 80.0; lim_rel_a = std::exp(-1.0 / (((rel_ms/1000.0)*m_sr) + 1e-9));
        lim_ga = 1.0;
        active_config.cf_lpL.set_lp(cf_cutoff, m_sr);
        active_config.cf_lpR.set_lp(cf_cutoff, m_sr);
        shadow_config.cf_lpL.set_lp(cf_cutoff, m_sr);
        shadow_config.cf_lpR.set_lp(cf_cutoff, m_sr);

        alpha_smooth.setup(10.0, m_sr);
        grain_smooth.setup(10.0, m_sr);
        ot_smooth.setup(10.0, m_sr);
        gain_smoother.setup(50.0, m_sr);

        // Poisson & FDN
        npp.setup(m_sr, (uint32_t)seed ^ 0xDEADBEEFu);
        active_config.fdn.setup(m_sr, 29.7, 37.1, 41.3, 43.9, 0.6);
        shadow_config.fdn.setup(m_sr, 29.7, 37.1, 41.3, 43.9, 0.6);

        // grain scheduling
        grain_count = 0;
        // YIN smoothing
        f0_hist[0]=f0_hist[1]=f0_hist[2]=2.0; f0_idx=0;

        return {};
    } };

    // setters per derived
    attribute<number>::setter lambda_setters = [this](number v){
        if(m_sr>0){
            shadow_config = active_config;  // Copy current
            shadow_config.prepare(m_sr);    // Update shadow
            shadow_config.yin.thresh = yin_thresh;
            shadow_config.yin.prepare(dft.N);
            shadow_config.ot.prepare(dft.N);
            shadow_config.fdn.setup(m_sr, 29.7, 37.1, 41.3, 43.9, 0.6);
            shadow_config.cf_lpL.set_lp(cf_cutoff, m_sr);
            shadow_config.cf_lpR.set_lp(cf_cutoff, m_sr);
            config_dirty = true;            // Signal swap needed
        }
        return v;
    };

    organic_noise_tilde(){
        mix.set_setter(lambda_setters); color.set_setter(lambda_setters); fbm_depth.set_setter(lambda_setters);
        ou_rate.set_setter(lambda_setters); ou_sigma.set_setter(lambda_setters); chaos.set_setter(lambda_setters);
        glitch_rate.set_setter(lambda_setters); glitch_depth.set_setter(lambda_setters); env_sense.set_setter(lambda_setters);
        emdr_rate.set_setter(lambda_setters); emdr_depth.set_setter(lambda_setters); seed.set_setter(lambda_setters);

        smr_alpha.set_setter(lambda_setters); grain_width.set_setter(lambda_setters); grain_rate.set_setter(lambda_setters);
        harmonicity.set_setter(lambda_setters); gate_thresh.set_setter(lambda_setters); phi_dither.set_setter(lambda_setters);

        harmonia.set_setter(lambda_setters); breathe_rate.set_setter(lambda_setters); breathe_depth.set_setter(lambda_setters);
        crossfeed.set_setter(lambda_setters); cf_cutoff.set_setter(lambda_setters); tp_ceiling.set_setter(lambda_setters);

        beta_1f.set_setter(lambda_setters); beta_mix.set_setter(lambda_setters);
        poisson_base.set_setter(lambda_setters); poisson_envamt.set_setter(lambda_setters);
        ot_enable.set_setter(lambda_setters); ot_tau.set_setter(lambda_setters);
        yin_enable.set_setter(lambda_setters); yin_thresh.set_setter(lambda_setters);
    }

    void operator()(audio_bundle_input& in, audio_bundle_output& out) {
        const double* in1p = in.samples(0); double* L = out.samples(0); double* R = out.samples(1); auto vs = in.frame_count();
        const bool  use_smr = (mode == symbol{"smr"});
        const double cf_v   = (double)crossfeed;

        for(size_t i=0;i<vs;i++){
            const double x = in1p ? in1p[i] : 0.0; // mono input

            // Envelope
            const double e = env.process(x) * env_sense;

            // ---------- Respirazione HARMONIA ----------
            breathe_phase += std::max(0.0001, (double)breathe_rate) * inv_sr;
            if(breathe_phase>=1.0) breathe_phase-=1.0;
            const double breath = 0.5 - 0.5*std::cos(2.0*PI*breathe_phase); // [0..1]
            const double d_breath = (double)breathe_depth;

            // ---------- CORE time-domain ----------
            double white = rng.bipolar();
            double pinkn = pink.process();
            double brownn = brown.process(white);
            double fracn = frac.process();

            double c = color;
            double np = (c<0.5) ? mix_lin(white,pinkn,c*2.0) : mix_lin(pinkn,brownn,(c-0.5)*2.0);
            np = mix_lin(np, fracn, beta_mix);

            double ouv = (ou.process()+1.0)*0.5;
            double bright = clamp01(0.2 + 0.6*ouv + 0.4*clamp01(e));
            double fbm = fbm_lps[0].process(white)*0.6 + fbm_lps[1].process(white)*0.3 + fbm_lps[2].process(white)*0.1;
            double organic = np + fbm_depth*fbm + 0.35*ouv*np;

            lfo_glitch += glitch_rate*inv_sr; if(lfo_glitch>=1.0){ lfo_glitch-=1.0; gstate = lmap.step(); }
            double gate = (gstate > (0.8 - 0.6*chaos)) ? 1.0 : 0.0; double glitch_amt = 1.0 - glitch_depth*gate;

            double wet_core = softsat(organic * (0.6 + 0.6*bright)) * glitch_amt;

            // ---------- STFT scheduling fisso + Poisson/grain_rate per re-target ----------
            if (use_smr) push_input(x);

            // λ(t): env + (opzionale) respiro
            double lambda = std::max(0.0, (double)poisson_base + (double)poisson_envamt * clamp01(e));
            if (harmonia > 0.5) {
                double wob = (breath - 0.5) * 2.0; // [-1,1]
                lambda *= (1.0 + 0.2 * d_breath * wob);
            }
            if(npp.event(lambda)) need_reloc = true;

            // reloc periodico da grain_rate
            int grain_period = (int)std::round(m_sr / std::max(5.0, (double)grain_rate));
            if (++grain_count >= std::max(1, grain_period)) { grain_count = 0; need_reloc = true; }

            // Esegui STFT ogni H campioni
            double wet_smr = wet_core;
            if(use_smr && ++hop_count >= dft.H){
                hop_count = 0;

                // Safe to swap configs at frame boundary
                if(config_dirty.load(std::memory_order_acquire)){
                    std::swap(active_config, shadow_config);
                    config_dirty.store(false, std::memory_order_release);
                }

                // Apply parameter smoothing
                const double local_alpha = alpha_smooth.process(smr_alpha);
                const double local_grain_width = grain_smooth.process(grain_width);
                const double local_ot_tau = ot_smooth.process(ot_tau);

                // copia frame
                copy_recent_to_frame(active_config.winbuf);

                // DFT
                dft.dft(active_config.winbuf, active_config.re, active_config.im);
                for(int k=0;k<dft.N;k++){
                    active_config.mag[k]=std::hypot(active_config.re[k], active_config.im[k]);
                    active_config.phase[k]=std::atan2(active_config.im[k], active_config.re[k]);
                }

                // α modulato dal respiro
                const double wob = (harmonia>0.5) ? (0.15 * d_breath * (breath - 0.5)) : 0.0;  // ±7.5%
                const double sigma = local_grain_width;
                const double b_inh = harmonicity;

                // f0 con YIN (τ→bin) + median(3)
                double f0_bin = 2.0;
                if (yin_enable > 0.5) {
                    // Use longer window for YIN
                    dft.copy_long_window(inring, in_wr, active_config.yin_long_win, 2); // Stride=2 for basic decimation
                    double mean = 0.0;
                    for(double v : active_config.yin_long_win) mean += v;
                    mean /= (double)active_config.yin_long_win.size();
                    for(size_t n=0; n<active_config.yin_long_win.size(); ++n)
                        active_config.xw[n] = active_config.yin_long_win[n] - mean;
                    double tau = active_config.yin.estimate_tau(active_config.xw);
                    if (!(tau >= 2.0 && tau < active_config.yin_long_win.size()/2)) tau = 2.0;
                    f0_bin = ((double)dft.N * m_sr) / (tau * 2.0 * 48000.0); // Adjust for decimation
                } else {
                    int kmax=2; double vmax=0.0; for(int k=2;k<dft.N/8;k++){ if(active_config.mag[k]>vmax){ vmax=active_config.mag[k]; kmax=k; } } f0_bin = std::max(2, kmax);
                }
                // Median(3)
                f0_hist[f0_idx] = f0_bin; f0_idx = (f0_idx+1) % 3;
                double a=f0_hist[0], b=f0_hist[1], c_=f0_hist[2];
                double lo = std::min(a, std::min(b, c_));
                double hi = std::max(a, std::max(b, c_));
                double f0_med = (a + b + c_) - lo - hi;
                f0_bin = std::clamp(f0_med, 2.0, (double)dft.N/2);

                // costruiamo GRANO spettrale
                std::fill(active_config.gr_re.begin(), active_config.gr_re.end(), 0.0);
                std::fill(active_config.gr_im.begin(), active_config.gr_im.end(), 0.0);

                for(int n=1;n<=10;n++){
                    double kn = f0_bin * n * std::sqrt(1.0 + b_inh*n*n);
                    if(kn<1 || kn>dft.N-2) continue;
                    double jitter = (need_reloc ? (rng.bipolar()*phi_dither) : 0.0) * dft.N;
                    double center = kn + jitter;
                    int k0 = std::max(1, (int)std::floor(center - 4*sigma));
                    int k1 = std::min(dft.N-2, (int)std::ceil(center + 4*sigma));
                    double amp = (1.0/n);
                    for(int k=k0;k<=k1;k++){
                        double w = std::exp(-0.5 * ((k-center)*(k-center)) / (sigma*sigma));
                        double m = amp * w;
                        active_config.gr_re[k] += m * std::cos(active_config.phase[k]);
                        active_config.gr_im[k] += m * std::sin(active_config.phase[k]);
                    }
                }
                need_reloc = false;

                // cleaning dolce
                const double tgh = gate_thresh;
                for(int k=0;k<dft.N;k++){
                    double ref = active_config.mag[k] + 1e-12;
                    double nrm = std::tanh(2.0*std::sqrt(ref));
                    double g = (nrm<=tgh) ? 0.2 * (nrm/tgh)
                                          : ([&](){ double u = (nrm - tgh) / std::max(1e-6, (1.0 - tgh)); u=clamp01(u); return u*u*(3.0 - 2.0*u); })();
                    active_config.gr_re[k]*=g; active_config.gr_im[k]*=g;
                }

                // OT barycenter (1D) tra |X| e |G| (massa conservata)
                if(ot_enable>0.5) {
                    double sumX = 0.0, sumG = 0.0;
                    for(int k=0; k<dft.N; k++) {
                        active_config.Xmag[k] = active_config.mag[k];
                        double gk = std::hypot(active_config.gr_re[k], active_config.gr_im[k]);
                        active_config.Gmag[k] = gk;
                        sumX += active_config.Xmag[k];
                        sumG += gk;
                    }
                    double S = gain_smoother.process((1.0 - local_alpha)*sumX + local_alpha*sumG);

                    active_config.ot.barycenter_push(active_config.Xmag, active_config.Gmag,
                                                     active_config.B, clamp01((double)local_ot_tau));

                    for(int k=0; k<dft.N; k++) {
                        double r2 = active_config.re[k]*active_config.re[k] + active_config.im[k]*active_config.im[k];
                        double invmag = (r2 > 1e-24) ? 1.0/std::sqrt(r2) : 0.0;
                        double u_re = active_config.re[k] * invmag;
                        double u_im = active_config.im[k] * invmag;
                        double mB   = S * active_config.B[k];
                        active_config.gr_re[k] = mB * u_re;
                        active_config.gr_im[k] = mB * u_im;
                    }
                }

                // Blend complesso
                for(int k=0;k<dft.N;k++){
                    active_config.re[k] = (1.0-local_alpha)*active_config.re[k] + local_alpha*active_config.gr_re[k];
                    active_config.im[k] = (1.0-local_alpha)*active_config.im[k] + local_alpha*active_config.gr_im[k];
                }

                // IFFT + OLA (H=N/2)
                dft.idft(active_config.re, active_config.im, active_config.yframe);
                add_ola(active_config.yframe);
            }

            // pull OLA ad ogni sample (solo se smr)
            if(use_smr){
                double wet = pull_ola();
                if(!std::isfinite(wet)) wet = 0.0;
                wet_smr = wet;
            }

            // Blend dry/wet globale
            double y = mix_lin(x, (use_smr)?wet_smr:wet_core, clamp01((double)mix));

            // micro-FDN "nuvola" (mix basso)
            const double fdn_send = 0.08;
            double y_fdn = active_config.fdn.tick(y);
            y = mix_lin(y, y_fdn, fdn_send);

            // EMDR L/R (range rilassante, senza scrivere attributi in audio thread)
            const double emdr_rate_l = std::min(1.4, std::max(0.8, (double)emdr_rate));
            const double emdr_depth_l = std::min(0.35, (double)emdr_depth);
            phase_emdr += (emdr_rate_l * inv_sr); if(phase_emdr>=1.0) phase_emdr-=1.0; double hemi = (phase_emdr<0.5)?1.0:-1.0;
            double gL = 1.0 - emdr_depth_l * ((hemi<0)?1.0:0.0);
            double gR = 1.0 - emdr_depth_l * ((hemi>0)?1.0:0.0);

            double yl = y * gL;
            double yr = y * gR;

            // Crossfeed cuffie (lowpass opposto)
            if (cf_v > 0.0001) {
                double cf = clamp01(cf_v);
                // HP the direct path (via LP subtraction)
                yl = yl - active_config.cf_hpL.process(yl);
                yr = yr - active_config.cf_hpR.process(yr);
                // LP the crossed path
                double addL = active_config.cf_lpR.process(yr);
                double addR = active_config.cf_lpL.process(yl);
                yl = yl + cf * addL * 0.5;
                yr = yr + cf * addR * 0.5;
            }

            // Soft true‑peak ceiling approx (thread-safe; fast-attack + soft release)
            auto soft_ceiling_stereo = [&](double &l, double &r) {
                const double al = std::fabs(l), ar = std::fabs(r);
                const double a = std::max(al, ar);
                if (a <= tp_lin) return;

                const double over = a - tp_lin;
                const double g = 1.0 / (1.0 + 4.0*over);
                lim_ga = std::min(lim_ga, g);
                lim_ga = lim_ga * lim_rel_a + g * (1.0 - lim_rel_a);

                auto apply = [&](double s) {
                    const double as = std::fabs(s);
                    if (as <= tp_lin) return s;
                    return std::copysign(tp_lin + (as - tp_lin)*lim_ga, s);
                };
                l = apply(l); r = apply(r);
            };
            soft_ceiling_stereo(yl, yr);

            if (!std::isfinite(yl)) yl = 0.0;
            if (!std::isfinite(yr)) yr = 0.0;
            L[i] = yl;
            R[i] = yr;
        }
    }

private:
    // state
    double m_sr {48000.0}; double inv_sr {1.0/48000.0};

    // modules
    XorShift32 rng; PinkVoss pink; Brown brown; OU ou; EnvFollow env; OnePole tilt_filter; OnePole fbm_lps[3]; DCBlock dc; Logistic lmap;

    // core state
    double phase_emdr {0.0}; double lfo_glitch {0.0}; double gstate {0.0};

    // STFT/OLA
    TinyDFT dft {128,64};
    int hop_count {0};
    bool need_reloc {false};

    // Config
    struct Config {
        std::vector<double> yin_long_win{std::vector<double>(384, 0.0)};
        std::vector<double> winbuf, re, im, yframe, mag, phase;
        std::vector<double> gr_re, gr_im;
        std::vector<double> xw, Xmag, Gmag, B;
        OnePole cf_lpL, cf_lpR;
        OnePole cf_hpL, cf_hpR;  // New HP for direct path
        FDN4 fdn;
        Yin yin;
        OT1D ot;
        
        void prepare(double sr) {
            // Initialize filters
            cf_lpL.set_lp(700.0, sr);
            cf_lpR.set_lp(700.0, sr);
            cf_hpL.set_lp(150.0, sr);  // HP via subtraction
            cf_hpR.set_lp(150.0, sr);
            
            // Preallocate vectors
            const int N = 128;
            winbuf.assign(N,0.0);
            re.assign(N,0.0); im.assign(N,0.0);
            yframe.assign(N,0.0);
            mag.assign(N,0.0); phase.assign(N,0.0);
            gr_re.assign(N,0.0); gr_im.assign(N,0.0);
            xw.resize(yin_long_win.size());
            Xmag.resize(N); Gmag.resize(N); B.resize(N);
            
            // Setup components
            yin.prepare(N);
            ot.prepare(N);
            fdn.setup(sr);
        }
    };

    std::atomic<bool> config_dirty{false};
    Config active_config;
    Config shadow_config;
    ParamSmoother alpha_smooth, grain_smooth, ot_smooth, gain_smoother;

    // I/O rings for STFT
    std::vector<double> inring{std::vector<double>(4096, 0.0)};
    int in_wr{0};
    std::vector<double> olaring{std::vector<double>(4096, 0.0)};
    int ola_rd{0}, ola_wr{0};

    // grain scheduling
    int    grain_count {0};
    // YIN smoothing
    double f0_hist[3]{2.0,2.0,2.0};
    int    f0_idx{0};

    // helpers ring/OLA
    inline void push_input(double x){ inring[in_wr] = x; in_wr = (in_wr+1) % (int)inring.size(); }
    inline void copy_recent_to_frame(std::vector<double>& dst){ int N=dft.N; int idx = in_wr - N; if(idx<0) idx += (int)inring.size(); for(int n=0;n<N;n++){ dst[n]=inring[(idx+n) % (int)inring.size()]; } }
    inline void add_ola(const std::vector<double>& frm){ int Nloc=(int)frm.size(); int H=dft.H; for(int n=0;n<Nloc;n++){ int pos = (ola_wr + n) % (int)olaring.size(); olaring[pos] += frm[n]; } ola_wr = (ola_wr + H) % (int)olaring.size(); }
    inline double pull_ola(){ double y = undenorm(olaring[ola_rd]); olaring[ola_rd] = 0.0; ola_rd = (ola_rd+1) % (int)olaring.size(); return y; }
};

MIN_EXTERNAL(organic_noise_tilde);

/*
README (inline)
================
Project layout per min-devkit:

min-devkit/
  source/projects/organic.noise_tilde/
    organic.noise_tilde.cpp   <-- (questo file)
    CMakeLists.txt

CMakeLists.txt:
---------------------------------
cmake_minimum_required(VERSION 3.20)
project(organic.noise_tilde)
include(../../min-api/script/min-pretarget.cmake)
add_library(${PROJECT_NAME} MODULE organic.noise_tilde.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
include(../../min-api/script/min-posttarget.cmake)
---------------------------------

Patch Max (test rapido):
-----------------------
[loadbang]
 |                \
 |                 [message preset asmr_tingles]
 |                  |
[adc~] -> [organic.noise~ @mode smr @mix 0.65 @color 0.35
          @env_sense 0.9 @emdr_rate 1.0 @emdr_depth 0.3
          @smr_alpha 0.55 @grain_width 7.5 @grain_rate 48
          @harmonicity 0.003 @gate_thresh 0.14 @phi_dither 0.004
          @harmonia 1 @breathe_rate 0.10 @breathe_depth 0.6
          @crossfeed 0.12 @cf_cutoff 700 @tp_ceiling -1.0
          @beta 1.0 @beta_mix 0.25
          @poisson_base 20. @poisson_envamt 40. @ot_enable 1 @ot_tau 0.5
          @yin_enable 1 @yin_thresh 0.10]
 |
[dac~]

Note tecniche chiave:
---------------------
- IDFT: index twiddle con Nloc (robusto a refactor).
- OT: clamp numerico di q prima di inv_cdf → niente edge-case a q==1.
- OLA: pull_ola() con undenorm (gratis e sicuro).
- Dev: assert (debug) che ring OLA ≥ 4 frame.
- Resto invariato: YIN τ→bin + median(3), hop fisso H=64, COLA sqrt-Hann, nessuna alloc in callback, Poisson clamp, limiter per-istanza fast-attack.
*/