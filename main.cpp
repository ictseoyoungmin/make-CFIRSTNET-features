// main.cpp
// IR/Distance/Resistance feature extractor (standalone)
// - Presets: sky130 / asap7 / nangate45 / custom
// - Layer name normalization: m1 / M1 / met1 → met1
// - gzip(.sp.gz) & plain(.sp)
// - Shared payload to preserve cross-layer current/IR coupling
// - Optional auto-orientation (--auto-ori), stats/CSV dump

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <filesystem>

#include <boost/multi_array.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file.hpp>

using namespace std;

#define VERTICAL   1
#define HORIZONTAL 2

// ---------------- small utils ----------------
struct coordinate { int x{0}; int y{0}; };
static inline bool operator==(const coordinate& a, const coordinate& b){ return a.x==b.x && a.y==b.y; }

template<class T> static inline T clampT(T v, T lo, T hi){ return v<lo?lo:(v>hi?hi:v); }
static inline int  rint_div(float v, float gap){ return (int)std::floor(v/gap + 0.5f); }

static inline string tolower_copy(string s){ for(char& c:s) c=(char)std::tolower((unsigned char)c); return s; }
static inline bool starts_with(const string& s, const string& p){ return s.rfind(p,0)==0; }
static inline bool ends_with(const string& s, const string& suf){ return s.size()>=suf.size() && s.compare(s.size()-suf.size(), suf.size(), suf)==0; }

static inline string norm_layer(string s){
    s = tolower_copy(std::move(s));
    if (starts_with(s, "met")) return s;
    if (!s.empty() && s[0]=='m') return "met"+s.substr(1);
    return s;
}
static inline int layer_num(const string& s_norm){
    // s_norm like "met1"
    if (starts_with(s_norm, "met")){
        try { return stoi(s_norm.substr(3)); } catch(...) { return INT_MAX/4; }
    }
    return INT_MAX/3;
}

static vector<string_view> MultiSplitStringV2(string_view input, string_view delims){
    vector<string_view> out;
    const char* b = input.data();
    const char* p = b;
    const char* e = b + input.size();
    for(; p!=e; ++p){
        bool hit=false;
        for(char d: delims){ if (*p==d){ hit=true; break; } }
        if (hit){ if (p>b) out.emplace_back(b, p-b); b=p+1; }
    }
    if (p>b) out.emplace_back(b, p-b);
    return out;
}

struct wire {
    coordinate c1{}, c2{};
    float resistance{0.f};
    float resistance_per_unit{0.f};
};
struct h_wire { int h{0}, w1{0}, w2{0}; };
struct v_wire { int w{0}, h1{0}, h2{0}; };

struct NodePayload {
    std::vector<float> ir_drop;  // size = 2*M - 1
    float current_source{0.f};
    explicit NodePayload(int ch=0) : ir_drop(ch, 0.f) {}
};

struct node {
    coordinate c{};
    int R1{0}, R2{0};
    float current{0.f};
    float resistance{0.f};
    std::shared_ptr<NodePayload> payload; // shared among copies

    node() = default;
    explicit node(int ch) : payload(std::make_shared<NodePayload>(ch)) {}
};

static inline bool in_range(const coordinate& c, const wire& w){
    return w.c1.x<=c.x && c.x<=w.c2.x && w.c1.y<=c.y && c.y<=w.c2.y;
}
static void sort_node_vertical(vector<node>& v){
    sort(v.begin(), v.end(), [](auto& a, auto& b){
        if (a.c.y != b.c.y) return a.c.y < b.c.y;
        if (a.c.x != b.c.x) return a.c.x < b.c.x;
        return false;
    });
}
static void sort_node_horizontal(vector<node>& v){
    sort(v.begin(), v.end(), [](auto& a, auto& b){
        if (a.c.x != b.c.x) return a.c.x < b.c.x;
        if (a.c.y != b.c.y) return a.c.y < b.c.y;
        return false;
    });
}
static void sort_wire_vertical(vector<wire>& v){
    sort(v.begin(), v.end(), [](auto& a, auto& b){
        if (a.c1.y != b.c1.y) return a.c1.y < b.c1.y;
        if (a.c1.x != b.c1.x) return a.c1.x < b.c1.x;
        return false;
    });
}
static void sort_wire_horizontal(vector<wire>& v){
    sort(v.begin(), v.end(), [](auto& a, auto& b){
        if (a.c1.x != b.c1.x) return a.c1.x < b.c1.x;
        if (a.c1.y != b.c1.y) return a.c1.y < b.c1.y;
        return false;
    });
}

// ---------------- circuit ----------------
struct local_partition {
    bool v1{false}, v2{false};
    float resistance_per_unit{0.f};
    vector<node> nodes;
};

class Circuit {
public:
    std::map<string,int> metal_idx;     // "met1" -> 0
    std::map<string,int> via_idx;       // "met1met4" -> 0 (between consecutive)
    vector<int> orientation;            // per metal

    vector<vector<node>> Vmap_raw, Imap_raw;
    vector<vector<wire>> R_raw, Via_raw;

    vector<vector<node>> current_map, voltage_map;
    vector<vector<wire>> wire_map;
    vector<vector<local_partition>> part;

    int x_max=0, y_max=0;
    int H=0, W=0;
    int IR_CH=0; // 2*M - 1

    void set_orientation(const vector<string>& metals, const vector<int>& ori){
        if (metals.empty() || metals.size()!=ori.size())
            throw runtime_error("invalid metals/orientation");
        metal_idx.clear(); via_idx.clear(); orientation.clear();
        for (int i=0;i<(int)metals.size();++i){
            string k = norm_layer(metals[i]);
            if (metal_idx.count(k)) throw runtime_error("duplicate metal: "+k);
            metal_idx[k]=i;
            orientation.push_back(ori[i]);
        }
        for (int i=0;i<(int)metals.size()-1;++i){
            string k = norm_layer(metals[i]) + norm_layer(metals[i+1]);
            via_idx[k]=i;
        }
        IR_CH = 2*(int)metals.size()-1;
    }
    void set_size(int h, int w){ if (h<=0||w<=0) throw runtime_error("invalid H/W"); H=h; W=w; }

    void read_data(const string& path){
        R_raw.assign(metal_idx.size(), {});
        Via_raw.assign(via_idx.size(), {});
        Imap_raw.assign(metal_idx.size(), {});
        Vmap_raw.assign(metal_idx.size(), {});

        namespace io = boost::iostreams;
        auto handle_line = [&](const string& line){
            auto toks = MultiSplitStringV2(line, " _");
            if (toks.empty()) return;
            char head = toks[0].empty()?'\0':toks[0][0];
            if (head=='R'){
                if (toks.size()<10) return;
                wire w;
                w.c1.x = atoi(string(toks[3]).c_str());
                w.c1.y = atoi(string(toks[4]).c_str());
                w.c2.x = atoi(string(toks[7]).c_str());
                w.c2.y = atoi(string(toks[8]).c_str());
                w.resistance = (float)atof(string(toks[9]).c_str());
                string m1 = norm_layer(string(toks[2]));
                string m2 = norm_layer(string(toks[6]));
                if (m1!=m2){
                    if (stoi(m1.substr(3)) > stoi(m2.substr(3))) std::swap(w.c1,w.c2);
                    auto it = via_idx.find(m1+m2);
                    if (it!=via_idx.end()) Via_raw[it->second].push_back(w);
                }else if (w.c1.x != w.c2.x){
                    if (w.c1.x > w.c2.x) std::swap(w.c1,w.c2);
                    auto it = metal_idx.find(m1); if (it!=metal_idx.end()) R_raw[it->second].push_back(w);
                }else if (w.c1.y != w.c2.y){
                    if (w.c1.y > w.c2.y) std::swap(w.c1,w.c2);
                    auto it = metal_idx.find(m1); if (it!=metal_idx.end()) R_raw[it->second].push_back(w);
                }
                x_max = max(x_max, w.c2.x);
                y_max = max(y_max, w.c2.y);
            }else if (head=='I'){
                if (toks.size()<7) return;
                node n(IR_CH);
                n.c.x = atoi(string(toks[3]).c_str());
                n.c.y = atoi(string(toks[4]).c_str());
                n.payload->current_source = (float)atof(string(toks[6]).c_str());
                string m1 = norm_layer(string(toks[2]));
                auto it = metal_idx.find(m1);
                if (it!=metal_idx.end()) Imap_raw[it->second].push_back(std::move(n));
            }else if (head=='V'){
                if (toks.size()<5) return;
                node n(IR_CH);
                n.c.x = atoi(string(toks[3]).c_str());
                n.c.y = atoi(string(toks[4]).c_str());
                string m1 = norm_layer(string(toks[2]));
                auto it = metal_idx.find(m1);
                if (it!=metal_idx.end()) Vmap_raw[it->second].push_back(std::move(n));
            }
        };

        if (ends_with(path, ".gz")){
            io::filtering_istream in;
            in.push(io::gzip_decompressor());
            in.push(io::file_source(path));
            if (!in.good()) throw runtime_error("open gz failed: "+path);
            string line; size_t cnt=0;
            while (getline(in,line)){ if(!line.empty()) handle_line(line); ++cnt; }
            if (cnt==0) throw runtime_error("empty gz file");
        }else{
            ifstream in(path);
            if (!in) throw runtime_error("open failed: "+path);
            string line; size_t cnt=0;
            while (getline(in,line)){ if(!line.empty()) handle_line(line); ++cnt; }
            if (cnt==0) throw runtime_error("empty file");
        }
        if (x_max<=0 || y_max<=0) throw runtime_error("parsed bounds are zero; check file");
    }

    void merge_wires(){
        wire_map.assign(R_raw.size(), {});
        for (int m=0; m<(int)metal_idx.size(); ++m){
            auto& r = R_raw[m];
            if (r.empty()){ wire_map[m].clear(); continue; }
            if (orientation[m]==VERTICAL) sort_wire_vertical(r);
            else                           sort_wire_horizontal(r);
            vector<wire> nw;
            wire w=r[0];
            for (int i=1; i<(int)r.size(); ++i){
                if (w.c2==r[i].c1){
                    w.c2=r[i].c2; w.resistance+=r[i].resistance;
                }else{
                    float len = (float)(w.c2.x-w.c1.x + w.c2.y-w.c1.y);
                    w.resistance_per_unit = len>0? (w.resistance/len) : 0.f;
                    nw.push_back(w); w=r[i];
                }
            }
            { float len = (float)(w.c2.x-w.c1.x + w.c2.y-w.c1.y);
              w.resistance_per_unit = len>0? (w.resistance/len) : 0.f;
              nw.push_back(w); }
            wire_map[m] = std::move(nw);
        }
    }

    void merge_nodes(){
        current_map.assign(metal_idx.size(), {});
        for (int m=0;m<(int)metal_idx.size();++m)
            for (auto n: Imap_raw[m]) current_map[m].push_back(std::move(n));

        // vias become nodes placed to (vi+1)-th metal (original rule)
        for (int vi=0; vi<(int)via_idx.size(); ++vi){
            for (auto& w: Via_raw[vi]){
                node n(IR_CH);
                n.c = (vi%2==0)? w.c2 : w.c1;
                n.resistance = w.resistance;
                int target_metal = vi+1;
                if (0<=target_metal && target_metal<(int)metal_idx.size())
                    current_map[target_metal].push_back(std::move(n));
            }
        }
        voltage_map = Vmap_raw;
        for (int m=0; m<(int)metal_idx.size(); ++m){
            if (orientation[m]==VERTICAL) sort_node_vertical(current_map[m]);
            else                           sort_node_horizontal(current_map[m]);
        }
    }

    void build_partitions(){
        part.assign(metal_idx.size(), {});
        if (wire_map.empty()) return;

        for (int m=(int)metal_idx.size()-1; m>=0; --m){
            auto& wires  = wire_map[m];
            auto& cnodes = current_map[m];

            bool isV=false; int ci=0, vi=0;

            if (orientation[m]==VERTICAL){
                sort_node_vertical(voltage_map[m]);
                auto& vnodes = voltage_map[m];

                for (auto& w: wires){
                    while (true){
                        if ((int)vnodes.size()==vi && (int)cnodes.size()==ci) break;
                        node n;
                        if ((int)vnodes.size()==vi){ n=cnodes[ci]; isV=false; }
                        else if ((int)cnodes.size()==ci){ n=vnodes[vi]; isV=true; }
                        else if (vnodes[vi].c.y < cnodes[ci].c.y){ n=vnodes[vi]; isV=true; }
                        else if (vnodes[vi].c.y > cnodes[ci].c.y){ n=cnodes[ci]; isV=false; }
                        else { if (vnodes[vi].c.x < cnodes[ci].c.x){ n=vnodes[vi]; isV=true; } else { n=cnodes[ci]; isV=false; } }

                        if (!in_range(n.c, w)){
                            if (n.c.y < w.c1.y || (n.c.y==w.c1.y && n.c.x < w.c1.x)){ if (isV) ++vi; else ++ci; continue; }
                            else break;
                        }
                        local_partition p; p.resistance_per_unit=w.resistance_per_unit; p.v1=isV;
                        while (true){
                            p.nodes.push_back(n); p.v2=isV;
                            if (isV) ++vi; else ++ci;
                            bool vend=((int)vnodes.size()==vi), cend=((int)cnodes.size()==ci);
                            if (vend && cend) break;
                            if (vend){ n=cnodes[ci]; isV=false; }
                            else if (cend){ n=vnodes[vi]; isV=true; }
                            else if (vnodes[vi].c.y < cnodes[ci].c.y){ n=vnodes[vi]; isV=true; }
                            else if (vnodes[vi].c.y > cnodes[ci].c.y){ n=cnodes[ci]; isV=false; }
                            else { if (vnodes[vi].c.x < cnodes[ci].c.x){ n=vnodes[vi]; isV=true; } else { n=cnodes[ci]; isV=false; } }
                            if (in_range(n.c,w)){ if (isV){ p.nodes.push_back(n); p.v2=isV; break; } }
                            else break;
                        }
                        if (p.nodes.size()>1 && (p.v1||p.v2)){
                            part[m].push_back(std::move(p));
                            if (m!=0){
                                auto& src = part[m].back().nodes;
                                int s = part[m].back().v1?1:0;
                                int e = (int)src.size() - (part[m].back().v2?1:0);
                                if (s<e) voltage_map[m-1].insert(voltage_map[m-1].end(), src.begin()+s, src.begin()+e);
                            }
                        }
                    }
                }
            }else{ // HORIZONTAL
                sort_node_horizontal(voltage_map[m]);
                auto& vnodes = voltage_map[m];

                for (auto& w: wires){
                    while (true){
                        if ((int)vnodes.size()==vi && (int)cnodes.size()==ci) break;
                        node n;
                        if ((int)vnodes.size()==vi){ n=cnodes[ci]; isV=false; }
                        else if ((int)cnodes.size()==ci){ n=vnodes[vi]; isV=true; }
                        else if (vnodes[vi].c.x < cnodes[ci].c.x){ n=vnodes[vi]; isV=true; }
                        else if (vnodes[vi].c.x > cnodes[ci].c.x){ n=cnodes[ci]; isV=false; }
                        else { if (vnodes[vi].c.y < cnodes[ci].c.y){ n=vnodes[vi]; isV=true; } else { n=cnodes[ci]; isV=false; } }

                        if (!in_range(n.c, w)){
                            if (n.c.x < w.c1.x || (n.c.x==w.c1.x && n.c.y < w.c1.y)){ if (isV) ++vi; else ++ci; continue; }
                            else break;
                        }
                        local_partition p; p.resistance_per_unit=w.resistance_per_unit; p.v1=isV;
                        while (true){
                            p.nodes.push_back(n); p.v2=isV;
                            if (isV) ++vi; else ++ci;
                            bool vend=((int)vnodes.size()==vi), cend=((int)cnodes.size()==ci);
                            if (vend && cend) break;
                            if (vend){ n=cnodes[ci]; isV=false; }
                            else if (cend){ n=vnodes[vi]; isV=true; }
                            else if (vnodes[vi].c.x < cnodes[ci].c.x){ n=vnodes[vi]; isV=true; }
                            else if (vnodes[vi].c.x > cnodes[ci].c.x){ n=cnodes[ci]; isV=false; }
                            else { if (vnodes[vi].c.y < cnodes[ci].c.y){ n=vnodes[vi]; isV=true; } else { n=cnodes[ci]; isV=false; } }
                            if (in_range(n.c, w)){ if (isV){ p.nodes.push_back(n); p.v2=isV; break; } }
                            else break;
                        }
                        if (p.nodes.size()>1 && (p.v1||p.v2)){
                            part[m].push_back(std::move(p));
                            if (m!=0){
                                auto& src = part[m].back().nodes;
                                int s = part[m].back().v1?1:0;
                                int e = (int)src.size() - (part[m].back().v2?1:0);
                                if (s<e) voltage_map[m-1].insert(voltage_map[m-1].end(), src.begin()+s, src.begin()+e);
                            }
                        }
                    }
                }
            }
        }
    }

    void compute_currents(){
        for (int m=0;m<(int)metal_idx.size();++m){
            auto& vec = part[m];
            if (vec.empty()) continue;

            if (orientation[m]==VERTICAL){
                for (auto& p: vec){
                    int x1=p.nodes.front().c.x, x2=p.nodes.back().c.x;
                    for (int i=(int)p.v1; i<(int)p.nodes.size()-(int)p.v2; ++i){
                        int R1=p.nodes[i].c.x-x1, R2=x2-p.nodes[i].c.x, R=R1+R2;
                        p.nodes[i].R1=R1; p.nodes[i].R2=R2;
                        float I1=0.f, I2=0.f;
                        if (p.v1 && !p.v2){ I1=p.nodes[i].payload->current_source; }
                        else if (!p.v1 && p.v2){ I2=p.nodes[i].payload->current_source; }
                        else if (R>0){ I1=p.nodes[i].payload->current_source*(float)R2/(float)R;
                                       I2=p.nodes[i].payload->current_source*(float)R1/(float)R; }
                        for (int j=0;j<=i;++j) p.nodes[j].current += I1;
                        for (int j=i+1;j<(int)p.nodes.size();++j) p.nodes[j].current -= I2;
                    }
                    if (p.v1) p.nodes.front().payload->current_source += std::abs(p.nodes.front().current);
                    if (p.v2) p.nodes.back(). payload->current_source += std::abs(p.nodes.back().current);
                }
            }else{
                for (auto& p: vec){
                    int y1=p.nodes.front().c.y, y2=p.nodes.back().c.y;
                    for (int i=(int)p.v1; i<(int)p.nodes.size()-(int)p.v2; ++i){
                        int R1=p.nodes[i].c.y-y1, R2=y2-p.nodes[i].c.y, R=R1+R2;
                        p.nodes[i].R1=R1; p.nodes[i].R2=R2;
                        float I1=0.f, I2=0.f;
                        if (p.v1 && !p.v2){ I1=p.nodes[i].payload->current_source; }
                        else if (!p.v1 && p.v2){ I2=p.nodes[i].payload->current_source; }
                        else if (R>0){ I1=p.nodes[i].payload->current_source*(float)R2/(float)R;
                                       I2=p.nodes[i].payload->current_source*(float)R1/(float)R; }
                        for (int j=0;j<=i;++j) p.nodes[j].current += I1;
                        for (int j=i+1;j<(int)p.nodes.size();++j) p.nodes[j].current -= I2;
                    }
                    if (p.v1) p.nodes.front().payload->current_source += std::abs(p.nodes.front().current);
                    if (p.v2) p.nodes.back(). payload->current_source += std::abs(p.nodes.back().current);
                }
            }
        }
    }

    void compute_ir_drop(){
        int M=(int)metal_idx.size();
        for (int m=M-1;m>=0;--m){
            auto& vec = part[m];
            if (vec.empty()) continue;

            if (orientation[m]==VERTICAL){
                for (auto& p: vec){
                    if (p.nodes.size()<2) continue;
                    if (p.v1 && p.v2){
                        auto& a = p.nodes.front().payload->ir_drop;
                        auto& b = p.nodes.back(). payload->ir_drop;
                        for (int i=1;i<(int)p.nodes.size()-1;++i){
                            p.nodes[i].payload->ir_drop[m*2] =
                                p.nodes[i-1].payload->ir_drop[m*2] +
                                p.nodes[i-1].current * (float)(p.nodes[i].c.x - p.nodes[i-1].c.x) * p.resistance_per_unit;
                            if (m!=0)
                                p.nodes[i].payload->ir_drop[m*2-1] =
                                    p.nodes[i].payload->current_source * p.nodes[i].resistance;
                            float denom=(float)(p.nodes[i].R1+p.nodes[i].R2);
                            float W = denom>0.f? (float)p.nodes[i].R1/denom : 0.f;
                            for (int mm=M-1; mm>m; --mm){
                                p.nodes[i].payload->ir_drop[mm*2]   = a[mm*2]   + W*(b[mm*2]-a[mm*2]);
                                p.nodes[i].payload->ir_drop[mm*2-1] = a[mm*2-1] + W*(b[mm*2-1]-a[mm*2-1]);
                            }
                        }
                    }else if (p.v1){
                        auto& a = p.nodes.front().payload->ir_drop;
                        for (int i=1;i<(int)p.nodes.size();++i){
                            p.nodes[i].payload->ir_drop[m*2] =
                                p.nodes[i-1].payload->ir_drop[m*2] +
                                p.nodes[i-1].current * (float)(p.nodes[i].c.x - p.nodes[i-1].c.x) * p.resistance_per_unit;
                            if (m!=0)
                                p.nodes[i].payload->ir_drop[m*2-1] =
                                    p.nodes[i].payload->current_source * p.nodes[i].resistance;
                            for (int mm=M-1; mm>m; --mm){
                                p.nodes[i].payload->ir_drop[mm*2]   = a[mm*2];
                                p.nodes[i].payload->ir_drop[mm*2-1] = a[mm*2-1];
                            }
                        }
                    }else if (p.v2){
                        auto& b = p.nodes.back().payload->ir_drop;
                        for (int i=(int)p.nodes.size()-2; i>=0; --i){
                            p.nodes[i].payload->ir_drop[m*2] =
                                p.nodes[i+1].payload->ir_drop[m*2] -
                                p.nodes[i+1].current * (float)(p.nodes[i+1].c.x - p.nodes[i].c.x) * p.resistance_per_unit;
                            if (m!=0)
                                p.nodes[i].payload->ir_drop[m*2-1] =
                                    p.nodes[i].payload->current_source * p.nodes[i].resistance;
                            for (int mm=M-1; mm>m; --mm){
                                p.nodes[i].payload->ir_drop[mm*2]   = b[mm*2];
                                p.nodes[i].payload->ir_drop[mm*2-1] = b[mm*2-1];
                            }
                        }
                    }
                }
            }else{ // HORIZONTAL
                for (auto& p: vec){
                    if (p.nodes.size()<2) continue;
                    if (p.v1 && p.v2){
                        auto& a = p.nodes.front().payload->ir_drop;
                        auto& b = p.nodes.back(). payload->ir_drop;
                        for (int i=1;i<(int)p.nodes.size()-1;++i){
                            p.nodes[i].payload->ir_drop[m*2] =
                                p.nodes[i-1].payload->ir_drop[m*2] +
                                p.nodes[i-1].current * (float)(p.nodes[i].c.y - p.nodes[i-1].c.y) * p.resistance_per_unit;
                            if (m!=0)
                                p.nodes[i].payload->ir_drop[m*2-1] =
                                    p.nodes[i].payload->current_source * p.nodes[i].resistance;
                            float denom=(float)(p.nodes[i].R1+p.nodes[i].R2);
                            float W = denom>0.f? (float)p.nodes[i].R1/denom : 0.f;
                            for (int mm=M-1; mm>m; --mm){
                                p.nodes[i].payload->ir_drop[mm*2]   = a[mm*2]   + W*(b[mm*2]-a[mm*2]);
                                p.nodes[i].payload->ir_drop[mm*2-1] = a[mm*2-1] + W*(b[mm*2-1]-a[mm*2-1]);
                            }
                        }
                    }else if (p.v1){
                        auto& a = p.nodes.front().payload->ir_drop;
                        for (int i=1;i<(int)p.nodes.size();++i){
                            p.nodes[i].payload->ir_drop[m*2] =
                                p.nodes[i-1].payload->ir_drop[m*2] +
                                p.nodes[i-1].current * (float)(p.nodes[i].c.y - p.nodes[i-1].c.y) * p.resistance_per_unit;
                            if (m!=0)
                                p.nodes[i].payload->ir_drop[m*2-1] =
                                    p.nodes[i].payload->current_source * p.nodes[i].resistance;
                            for (int mm=M-1; mm>m; --mm){
                                p.nodes[i].payload->ir_drop[mm*2]   = a[mm*2];
                                p.nodes[i].payload->ir_drop[mm*2-1] = a[mm*2-1];
                            }
                        }
                    }else if (p.v2){
                        auto& b = p.nodes.back().payload->ir_drop;
                        for (int i=(int)p.nodes.size()-2; i>=0; --i){
                            p.nodes[i].payload->ir_drop[m*2] =
                                p.nodes[i+1].payload->ir_drop[m*2] -
                                p.nodes[i+1].current * (float)(p.nodes[i+1].c.y - p.nodes[i].c.y) * p.resistance_per_unit;
                            if (m!=0)
                                p.nodes[i].payload->ir_drop[m*2-1] =
                                    p.nodes[i].payload->current_source * p.nodes[i].resistance;
                            for (int mm=M-1; mm>m; --mm){
                                p.nodes[i].payload->ir_drop[mm*2]   = b[mm*2];
                                p.nodes[i].payload->ir_drop[mm*2-1] = b[mm*2-1];
                            }
                        }
                    }
                }
            }
        }
    }

    vector<vector<node>> collect_ir_columns(){
        if (part.empty() || part[0].empty())
            throw runtime_error("no partitions on metal[0]");
        vector<vector<node>> out; vector<node> col;
        int prev_y = part[0].front().nodes.front().c.y;
        for (auto& p: part[0]){
            if (prev_y != p.nodes.front().c.y){
                if (!col.empty()) out.push_back(col);
                col.clear(); prev_y = p.nodes.front().c.y;
            }
            for (auto nd: p.nodes) col.push_back(nd);
        }
        if (!col.empty()) out.push_back(col);
        return out;
    }

    boost::multi_array<float,3> build_ir_map(const vector<vector<node>>& chip){
        int M=(int)metal_idx.size();
        int Vias=(int)via_idx.size();
        boost::multi_array<float,3> out(boost::extents[M+Vias][H][W]);

        float h_gap = (float)x_max/(float)H;
        float w_gap = (float)y_max/(float)W;

        vector<vector<vector<float>>> value_map;
        vector<float> y_map;

        // vertical integrate per column
        for (auto& c: chip){
            vector<vector<float>> values; values.reserve(H);
            int idx=0; float x1,x2,w,v1,v2;
            auto val=[&](int i,int ch){ return c[i].payload->ir_drop[ch]; };

            for (int h=0; h<H; ++h){
                vector<float> value((size_t)(M+Vias), 0.f);
                x1=h*h_gap; x2=x1+h_gap;
                idx = clampT(idx, 0, (int)c.size()-1);

                if (idx==(int)c.size()-1){
                    if (x1<=c[idx].c.x && c[idx].c.x<=x2){
                        for (int m=0;m<M+Vias;++m){
                            if (idx-1>=0){
                                float denom=(float)(c[idx].c.x - c[idx-1].c.x);
                                if (fabs(denom)>1e-6f){
                                    w=(val(idx,m)-val(idx-1,m))/denom;
                                    v1=(x1 - c[idx-1].c.x)*w + val(idx-1,m);
                                    value[m] += (v1 + val(idx,m)) * (c[idx].c.x - x1) * 0.5f;
                                    value[m] +=  val(idx,m) * (x2 - c[idx].c.x);
                                    value[m] /= h_gap;
                                }else value[m]=val(idx,m);
                            }else value[m]=val(idx,m);
                        }
                    }else if (x2<c[idx].c.x){
                        for (int m=0;m<M+Vias;++m){
                            if (idx-1>=0){
                                float denom=(float)(c[idx].c.x - c[idx-1].c.x);
                                if (fabs(denom)>1e-6f){
                                    w=(val(idx,m)-val(idx-1,m))/denom;
                                    v1=(x1 - c[idx-1].c.x)*w + val(idx-1,m);
                                    v2=(x2 - c[idx-1].c.x)*w + val(idx-1,m);
                                    value[m] = 0.5f*(v1+v2);
                                }else value[m]=val(idx,m);
                            }else value[m]=val(idx,m);
                        }
                    }else{
                        for (int m=0;m<M+Vias;++m) value[m]=val(idx,m);
                    }
                    values.push_back(value);
                    continue;
                }

                if (idx==0){
                    if (x1<=c[idx].c.x && c[idx].c.x<=x2){
                        for (int m=0;m<M+Vias;++m) value[m]+= val(idx,m) * (c[idx].c.x - x1);
                    }else{
                        for (int m=0;m<M+Vias;++m) value[m]=val(idx,m);
                        values.push_back(value); continue;
                    }
                }else{
                    if (x1<=c[idx].c.x && c[idx].c.x<=x2){
                        for (int m=0;m<M+Vias;++m){
                            float denom=(float)(c[idx].c.x - c[idx-1].c.x);
                            if (fabs(denom)>1e-6f){
                                w=(val(idx,m)-val(idx-1,m))/denom;
                                v1=(x1 - c[idx-1].c.x)*w + val(idx-1,m);
                                value[m] += (v1 + val(idx,m)) * (c[idx].c.x - x1) * 0.5f;
                            }else value[m] += val(idx,m)*(c[idx].c.x - x1);
                        }
                    }else{
                        for (int m=0;m<M+Vias;++m){
                            float denom=(float)(c[idx].c.x - c[idx-1].c.x);
                            if (fabs(denom)>1e-6f){
                                w=(val(idx,m)-val(idx-1,m))/denom;
                                v1=(x1 - c[idx-1].c.x)*w + val(idx-1,m);
                                v2=(x2 - c[idx-1].c.x)*w + val(idx-1,m);
                                value[m] = 0.5f*(v1+v2);
                            }else value[m]=val(idx,m);
                        }
                        values.push_back(value); continue;
                    }
                }

                while (x1<=c[idx].c.x && (idx+1)<(int)c.size() && c[idx+1].c.x<=x2){
                    for (int m=0;m<M+Vias;++m)
                        value[m] += (float)(c[idx+1].c.x - c[idx].c.x) * (val(idx+1,m)+val(idx,m)) * 0.5f;
                    ++idx; if (idx==(int)c.size()-1) break;
                }

                if (idx==(int)c.size()-1){
                    for (int m=0;m<M+Vias;++m) value[m] += val(idx,m) * (x2 - c[idx].c.x);
                }else{
                    for (int m=0;m<M+Vias;++m){
                        float denom=(float)(c[idx+1].c.x - c[idx].c.x);
                        if (fabs(denom)>1e-6f){
                            w=(val(idx+1,m)-val(idx,m))/denom;
                            v2=(x2 - c[idx].c.x)*w + val(idx,m);
                            value[m] += (val(idx,m)+v2)*(x2 - c[idx].c.x)*0.5f;
                        }else value[m] += val(idx,m)*(x2 - c[idx].c.x);
                    }
                }
                for (int m=0;m<M+Vias;++m) value[m]/=h_gap;
                values.push_back(value);
                if (idx!=(int)c.size()-1) ++idx;
            }
            value_map.push_back(std::move(values));
            y_map.push_back((float)c.front().c.y);
        }

        // horizontal integrate
        int idx=0; float y1,y2,w,v1,v2;
        auto val=[&](int col,int hh,int ch){ return value_map[col][hh][ch]; };

        for (int ww=0; ww<W; ++ww){
            y1=ww*w_gap; y2=y1+w_gap;
            idx = clampT(idx, 0, (int)y_map.size()-1);

            if (idx==(int)y_map.size()-1){
                if (y1<=y_map[idx] && y_map[idx]<=y2){
                    for (int h=0;h<H;++h)
                    for (int m=0;m<M+Vias;++m){
                        if (idx-1>=0){
                            float denom=(y_map[idx]-y_map[idx-1]);
                            if (fabs(denom)>1e-6f){
                                w=(val(idx,h,m)-val(idx-1,h,m))/denom;
                                v1=(y1 - y_map[idx-1])*w + val(idx-1,h,m);
                                out[m][h][ww] += (v1 + val(idx,h,m))*(y_map[idx]-y1)*0.5f;
                                out[m][h][ww] +=  val(idx,h,m)*(y2 - y_map[idx]);
                                out[m][h][ww] /= w_gap;
                            }else out[m][h][ww]=val(idx,h,m);
                        }else out[m][h][ww]=val(idx,h,m);
                    }
                }else if (y2<y_map[idx]){
                    for (int h=0;h<H;++h)
                    for (int m=0;m<M+Vias;++m){
                        if (idx-1>=0){
                            float denom=(y_map[idx]-y_map[idx-1]);
                            if (fabs(denom)>1e-6f){
                                w=(val(idx,h,m)-val(idx-1,h,m))/denom;
                                v1=(y1 - y_map[idx-1])*w + val(idx-1,h,m);
                                v2=(y2 - y_map[idx-1])*w + val(idx-1,h,m);
                                out[m][h][ww] = 0.5f*(v1+v2);
                            }else out[m][h][ww]=val(idx,h,m);
                        }else out[m][h][ww]=val(idx,h,m);
                    }
                }else{
                    for (int h=0;h<H;++h)
                    for (int m=0;m<M+Vias;++m) out[m][h][ww]=val(idx,h,m);
                }
                continue;
            }

            if (idx==0){
                if (y1<=y_map[idx] && y_map[idx]<=y2){
                    for (int h=0;h<H;++h)
                    for (int m=0;m<M+Vias;++m)
                        out[m][h][ww] += val(idx,h,m)*(y_map[idx]-y1);
                }else{
                    for (int h=0;h<H;++h)
                    for (int m=0;m<M+Vias;++m)
                        out[m][h][ww] = val(idx,h,m);
                    continue;
                }
            }else{
                if (y1<=y_map[idx] && y_map[idx]<=y2){
                    for (int h=0;h<H;++h)
                    for (int m=0;m<M+Vias;++m){
                        float denom=(y_map[idx]-y_map[idx-1]);
                        if (fabs(denom)>1e-6f){
                            w=(val(idx,h,m)-val(idx-1,h,m))/denom;
                            v1=(y1 - y_map[idx-1])*w + val(idx-1,h,m);
                            out[m][h][ww] += (v1 + val(idx,h,m))*(y_map[idx]-y1)*0.5f;
                        }else out[m][h][ww] += val(idx,h,m)*(y_map[idx]-y1);
                    }
                }else{
                    for (int h=0;h<H;++h)
                    for (int m=0;m<M+Vias;++m){
                        float denom=(y_map[idx]-y_map[idx-1]);
                        if (fabs(denom)>1e-6f){
                            w=(val(idx,h,m)-val(idx-1,h,m))/denom;
                            v1=(y1 - y_map[idx-1])*w + val(idx-1,h,m);
                            v2=(y2 - y_map[idx-1])*w + val(idx-1,h,m);
                            out[m][h][ww] = 0.5f*(v1+v2);
                        }else out[m][h][ww] = val(idx,h,m);
                    }
                    continue;
                }
            }

            while (y1<=y_map[idx] && (idx+1)<(int)y_map.size() && y_map[idx+1]<=y2){
                for (int h=0;h<H;++h)
                for (int m=0;m<M+Vias;++m)
                    out[m][h][ww] += (val(idx+1,h,m)+val(idx,h,m))*(y_map[idx+1]-y_map[idx])*0.5f;
                ++idx; if (idx==(int)y_map.size()-1) break;
            }

            if (idx==(int)y_map.size()-1){
                for (int h=0;h<H;++h)
                for (int m=0;m<M+Vias;++m)
                    out[m][h][ww] += val(idx,h,m)*(y2 - y_map[idx]);
            }else{
                for (int h=0;h<H;++h)
                for (int m=0;m<M+Vias;++m){
                    float denom=(y_map[idx+1]-y_map[idx]);
                    if (fabs(denom)>1e-6f){
                        w=(val(idx+1,h,m)-val(idx,h,m))/denom;
                        v2=(y2 - y_map[idx])*w + val(idx,h,m);
                        out[m][h][ww] += (val(idx,h,m)+v2)*(y2 - y_map[idx])*0.5f;
                    }else out[m][h][ww] += val(idx,h,m)*(y2 - y_map[idx]);
                }
            }
            for (int h=0;h<H;++h)
                for (int m=0;m<M+Vias;++m)
                    out[m][h][ww] /= w_gap;

            if (idx!=(int)y_map.size()-1) ++idx;
        }
        return out;
    }

    boost::multi_array<float,3> build_distance(){
        int M=(int)metal_idx.size();
        if (M<3) return boost::multi_array<float,3>(boost::extents[0][H][W]);

        boost::multi_array<float,3> out(boost::extents[M*2-3][H][W]);
        float h_gap=(float)x_max/(float)H;
        float w_gap=(float)y_max/(float)W;

        vector<h_wire> h_wires,new_h,no_dist_h;
        vector<v_wire> v_wires,new_v,no_dist_v;

        int cnt=0;
        for (int m=1;m<M;++m){
            auto& ws = wire_map[m];

            if (orientation[m]==HORIZONTAL){
                h_wires.clear(); new_h.clear();
                h_wires.push_back({0,0,y_max});
                for (auto& wr: ws){
                    int wh1=wr.c1.x, ww1=wr.c1.y, ww2=wr.c2.y;
                    for (auto& hw: h_wires){
                        int h1=hw.h, w1=hw.w1, w2=hw.w2;
                        if (w2<ww1 || ww2<w1){
                            new_h.push_back({h1,w1,w2});
                        }else if (ww1<=w1 && w2<=ww2){
                            int hh1=clampT(rint_div(h1,h_gap),0,H);
                            int hh2=clampT((int)floor(wh1/h_gap+0.5f),0,H);
                            int wl =clampT(rint_div(w1,w_gap),0,W);
                            int wrx=clampT((int)floor(w2 /w_gap+0.5f),0,W);
                            for(int h=hh1;h<hh2;++h){ float d=h*h_gap-h1; for(int w=wl;w<wrx;++w) out[cnt][h][w]=d; }
                        }else if (w1<ww1 && ww2<w2){
                            int hh1=clampT(rint_div(h1,h_gap),0,H);
                            int hh2=clampT((int)floor(wh1/h_gap+0.5f),0,H);
                            int wl =clampT(rint_div(ww1,w_gap),0,W);
                            int wrx=clampT((int)floor(ww2/w_gap+0.5f),0,W);
                            for(int h=hh1;h<hh2;++h){ float d=h*h_gap-h1; for(int w=wl;w<wrx;++w) out[cnt][h][w]=d; }
                            new_h.push_back({h1,w1,ww1});
                            new_h.push_back({h1,ww2,w2});
                        }else if (ww2<w2){
                            int hh1=clampT(rint_div(h1,h_gap),0,H);
                            int hh2=clampT((int)floor(wh1/h_gap+0.5f),0,H);
                            int wl =clampT(rint_div(w1,w_gap),0,W);
                            int wrx=clampT((int)floor(ww2/w_gap+0.5f),0,W);
                            for(int h=hh1;h<hh2;++h){ float d=h*h_gap-h1; for(int w=wl;w<wrx;++w) out[cnt][h][w]=d; }
                            new_h.push_back({h1,ww2,w2});
                        }else{ // w1<ww1
                            int hh1=clampT(rint_div(h1,h_gap),0,H);
                            int hh2=clampT((int)floor(wh1/h_gap+0.5f),0,H);
                            int wl =clampT(rint_div(ww1,w_gap),0,W);
                            int wrx=clampT((int)floor(w2 /w_gap+0.5f),0,W);
                            for(int h=hh1;h<hh2;++h){ float d=h*h_gap-h1; for(int w=wl;w<wrx;++w) out[cnt][h][w]=d; }
                            new_h.push_back({h1,w1,ww1});
                        }
                    }
                    h_wires.swap(new_h); new_h.clear();
                }
                no_dist_h = h_wires;

                for (int idx=0; idx<(int)ws.size(); ++idx){
                    h_wires.clear(); new_h.clear();
                    h_wires.push_back({ws[idx].c1.x, ws[idx].c1.y, ws[idx].c2.y});
                    for (int i=idx+1;i<(int)ws.size();++i){
                        auto& wr=ws[i]; int wh1=wr.c1.x, ww1=wr.c1.y, ww2=wr.c2.y;
                        for (auto& hw: h_wires){
                            int h1=hw.h, w1=hw.w1, w2=hw.w2;
                            if (w2<ww1 || ww2<w1){
                                new_h.push_back({h1,w1,w2});
                            }else if (ww1<=w1 && w2<=ww2){
                                int hh1=clampT(rint_div(h1,h_gap),0,H);
                                int hh2=clampT((int)floor(wh1/h_gap+0.5f),0,H);
                                int wl =clampT(rint_div(w1,w_gap),0,W);
                                int wrx=clampT((int)floor(w2 /w_gap+0.5f),0,W);
                                for (int h=hh1;h<hh2;++h){ float d=min(h*h_gap-h1, (float)wh1-h*h_gap); for(int w=wl;w<wrx;++w) out[cnt][h][w]=d; }
                            }else if (w1<ww1 && ww2<w2){
                                int hh1=clampT(rint_div(h1,h_gap),0,H);
                                int hh2=clampT((int)floor(wh1/h_gap+0.5f),0,H);
                                int wl =clampT(rint_div(ww1,w_gap),0,W);
                                int wrx=clampT((int)floor(ww2/w_gap+0.5f),0,W);
                                for (int h=hh1;h<hh2;++h){ float d=min(h*h_gap-h1, (float)wh1-h*h_gap); for(int w=wl;w<wrx;++w) out[cnt][h][w]=d; }
                                new_h.push_back({h1,w1,ww1});
                                new_h.push_back({h1,ww2,w2});
                            }else if (ww2<w2){
                                int hh1=clampT(rint_div(h1,h_gap),0,H);
                                int hh2=clampT((int)floor(wh1/h_gap+0.5f),0,H);
                                int wl =clampT(rint_div(w1,w_gap),0,W);
                                int wrx=clampT((int)floor(ww2/w_gap+0.5f),0,W);
                                for (int h=hh1;h<hh2;++h){ float d=min(h*h_gap-h1, (float)wh1-h*h_gap); for(int w=wl;w<wrx;++w) out[cnt][h][w]=d; }
                                new_h.push_back({h1,ww2,w2});
                            }else{
                                int hh1=clampT(rint_div(h1,h_gap),0,H);
                                int hh2=clampT((int)floor(wh1/h_gap+0.5f),0,H);
                                int wl =clampT(rint_div(ww1,w_gap),0,W);
                                int wrx=clampT((int)floor(w2 /w_gap+0.5f),0,W);
                                for (int h=hh1;h<hh2;++h){ float d=min(h*h_gap-h1, (float)wh1-h*h_gap); for(int w=wl;w<wrx;++w) out[cnt][h][w]=d; }
                                new_h.push_back({h1,w1,ww1});
                            }
                        }
                        h_wires.swap(new_h); new_h.clear();
                    }
                    for (auto& hw: h_wires){
                        int h1=hw.h, w1=hw.w1, w2=hw.w2;
                        int hh1=clampT(rint_div(h1,h_gap),0,H);
                        int wl =clampT(rint_div(w1,w_gap),0,W);
                        int wrx=clampT((int)floor(w2/w_gap+0.5f),0,W);
                        for (int h=hh1;h<H;++h){ float d=h*h_gap-h1; for(int w=wl;w<wrx;++w) out[cnt][h][w]=d; }
                    }
                }

                for (auto& hw: no_dist_h){
                    int w1=hw.w1, w2=hw.w2;
                    int wl=clampT(rint_div(w1,w_gap),0,W);
                    int wrx=clampT((int)floor(w2/w_gap+0.5f),0,W);
                    for (int h=0;h<H;++h){
                        float d=0.f;
                        if (wl<=0 && wrx<W) d=out[cnt][h][wrx];
                        else if (wl>0)      d=out[cnt][h][wl-1];
                        for (int w=wl; w<wrx; ++w) out[cnt][h][w]=d;
                    }
                }
            }else{ // VERTICAL
                v_wires.clear(); new_v.clear();
                v_wires.push_back({0,0,x_max});
                for (auto& wr: ws){
                    int ww1=wr.c1.y, wh1=wr.c1.x, wh2=wr.c2.x;
                    for (auto& vw: v_wires){
                        int w1=vw.w, h1=vw.h1, h2=vw.h2;
                        if (h2<wh1 || wh2<h1){
                            new_v.push_back({w1,h1,h2});
                        }else if (wh1<=h1 && h2<=wh2){
                            int wl =clampT(rint_div(w1,w_gap),0,W);
                            int wrx=clampT((int)floor(ww1/w_gap+0.5f),0,W);
                            int hh1=clampT(rint_div(h1,h_gap),0,H);
                            int hh2=clampT((int)floor(h2 /h_gap+0.5f),0,H);
                            for (int w=wl;w<wrx;++w){ float d=(w+0.5f)*w_gap - w1; for (int h=hh1;h<hh2;++h) out[cnt][h][w]=d; }
                        }else if (h1<wh1 && wh2<h2){
                            int wl =clampT(rint_div(w1,w_gap),0,W);
                            int wrx=clampT((int)floor(ww1/w_gap+0.5f),0,W);
                            int hh1=clampT(rint_div(h1,h_gap),0,H);
                            int hh2=clampT((int)floor(h2 /h_gap+0.5f),0,H);
                            for (int w=wl;w<wrx;++w){ float d=(w+0.5f)*w_gap - w1; for (int h=hh1;h<hh2;++h) out[cnt][h][w]=d; }
                            new_v.push_back({w1,h1,wh1});
                            new_v.push_back({w1,wh2,h2});
                        }else if (wh2<h2){
                            int wl =clampT(rint_div(w1,w_gap),0,W);
                            int wrx=clampT((int)floor(ww1/w_gap+0.5f),0,W);
                            int hh1=clampT(rint_div(h1,h_gap),0,H);
                            int hh2=clampT((int)floor(wh2/h_gap+0.5f),0,H);
                            for (int w=wl;w<wrx;++w){ float d=(w+0.5f)*w_gap - w1; for (int h=hh1;h<hh2;++h) out[cnt][h][w]=d; }
                            new_v.push_back({w1,wh2,h2});
                        }else{
                            int wl =clampT(rint_div(w1,w_gap),0,W);
                            int wrx=clampT((int)floor(ww1/w_gap+0.5f),0,W);
                            int hh1=clampT(rint_div(wh1,h_gap),0,H);
                            int hh2=clampT((int)floor(h2 /h_gap+0.5f),0,H);
                            for (int w=wl;w<wrx;++w){ float d=(w+0.5f)*w_gap - w1; for (int h=hh1;h<hh2;++h) out[cnt][h][w]=d; }
                            new_v.push_back({w1,h1,wh1});
                        }
                    }
                    v_wires.swap(new_v); new_v.clear();
                }
                no_dist_v = v_wires;

                for (int idx=0; idx<(int)ws.size(); ++idx){
                    v_wires.clear(); new_v.clear();
                    v_wires.push_back({ws[idx].c1.y, ws[idx].c1.x, ws[idx].c2.x});
                    for (int i=idx+1;i<(int)ws.size();++i){
                        auto& wr=ws[i]; int ww1=wr.c1.y, wh1=wr.c1.x, wh2=wr.c2.x;
                        for (auto& vw: v_wires){
                            int w1=vw.w, h1=vw.h1, h2=vw.h2;
                            if (h2<wh1 || wh2<h1){
                                new_v.push_back({w1,h1,h2});
                            }else if (wh1<=h1 && h2<=wh2){
                                int wl =clampT(rint_div(w1,w_gap),0,W);
                                int wrx=clampT((int)floor(ww1/w_gap+0.5f),0,W);
                                int hh1=clampT(rint_div(h1,h_gap),0,H);
                                int hh2=clampT((int)floor(h2 /h_gap+0.5f),0,H);
                                for (int w=wl;w<wrx;++w){ float d=min((w+0.5f)*w_gap - w1, (float)ww1 - (w+0.5f)*w_gap);
                                    for (int h=hh1;h<hh2;++h) out[cnt][h][w]=d; }
                            }else if (h1<wh1 && wh2<h2){
                                int wl =clampT(rint_div(w1,w_gap),0,W);
                                int wrx=clampT((int)floor(ww1/w_gap+0.5f),0,W);
                                int hh1=clampT(rint_div(h1,h_gap),0,H);
                                int hh2=clampT((int)floor(h2 /h_gap+0.5f),0,H);
                                for (int w=wl;w<wrx;++w){ float d=min((w+0.5f)*w_gap - w1, (float)ww1 - (w+0.5f)*w_gap);
                                    for (int h=hh1;h<hh2;++h) out[cnt][h][w]=d; }
                                new_v.push_back({w1,h1,wh1});
                                new_v.push_back({w1,wh2,h2});
                            }else if (wh2<h2){
                                int wl =clampT(rint_div(w1,w_gap),0,W);
                                int wrx=clampT((int)floor(ww1/w_gap+0.5f),0,W);
                                int hh1=clampT(rint_div(h1,h_gap),0,H);
                                int hh2=clampT((int)floor(wh2/h_gap+0.5f),0,H);
                                for (int w=wl;w<wrx;++w){ float d=min((w+0.5f)*w_gap - w1, (float)ww1 - (w+0.5f)*w_gap);
                                    for (int h=hh1;h<hh2;++h) out[cnt][h][w]=d; }
                                new_v.push_back({w1,wh2,h2});
                            }else{
                                int wl =clampT(rint_div(w1,w_gap),0,W);
                                int wrx=clampT((int)floor(ww1/w_gap+0.5f),0,W);
                                int hh1=clampT(rint_div(wh1,h_gap),0,H);
                                int hh2=clampT((int)floor(h2 /h_gap+0.5f),0,H);
                                for (int w=wl;w<wrx;++w){ float d=min((w+0.5f)*w_gap - w1, (float)ww1 - (w+0.5f)*w_gap);
                                    for (int h=hh1;h<hh2;++h) out[cnt][h][w]=d; }
                                new_v.push_back({w1,h1,wh1});
                            }
                        }
                        v_wires.swap(new_v); new_v.clear();
                    }
                    for (auto& vw: v_wires){
                        int w1=vw.w, h1=vw.h1, h2=vw.h2;
                        int wl =clampT(rint_div(w1,w_gap),0,W);
                        int hh1=clampT(rint_div(h1,h_gap),0,H);
                        int hh2=clampT((int)floor(h2/h_gap+0.5f),0,H);
                        for (int w=wl; w<W; ++w){ float d=(w+0.5f)*w_gap - w1; for (int h=hh1;h<hh2;++h) out[cnt][h][w]=d; }
                    }
                }

                for (auto& vw: no_dist_v){
                    int h1=vw.h1, h2=vw.h2;
                    int hh1=clampT(rint_div(h1,h_gap),0,H);
                    int hh2=clampT((int)floor(h2/h_gap+0.5f),0,H);
                    for (int w=0; w<W; ++w){
                        float d=0.f;
                        if (hh1<=0 && hh2<H) d=out[cnt][hh2][w];
                        else if (hh1>0)      d=out[cnt][hh1-1][w];
                        for (int h=hh1; h<hh2; ++h) out[cnt][h][w]=d;
                    }
                }
            }

            ++cnt;
        }

        for (int m=0; m<M-2; ++m){
            for (int h=0;h<H;++h)
                for (int w=0; w<W; ++w)
                    out[cnt][h][w] = out[m][h][w] + out[m+1][h][w];
            ++cnt;
        }

        return out;
    }

    boost::multi_array<float,3> build_res_mask(){
        int M=(int)metal_idx.size();
        if (M<3) return boost::multi_array<float,3>(boost::extents[0][H][W]);

        boost::multi_array<float,3> out(boost::extents[2*M-3][H][W]);
        float h_gap=(float)x_max/(float)H;
        float w_gap=(float)y_max/(float)W;

        int cnt=0;
        for (int m=1;m<M;++m){
            auto& ws = wire_map[m];
            for (auto& w: ws){
                if (w.c1.x == w.c2.x){
                    int x1 = clampT((int)floor(w.c2.x/h_gap - 0.5f), 0, H-1);
                    int x2 = clampT(x1+1, 0, H-1);
                    float ww1 = x2 - (w.c2.x/h_gap - 0.5f);
                    float ww2 = (w.c2.x/h_gap - 0.5f) - x1;
                    int y1 = clampT((int)floor(w.c1.y/w_gap - 0.5f), 0, W-1);
                    int y2 = clampT((int)ceil (w.c2.y/w_gap - 0.5f),  0, W-1);
                    for (int i=y1;i<=y2;++i){
                        if (i>=0 && i<W && x1>=0 && x1<H) out[cnt][x1][i]=ww1;
                        if (i>=0 && i<W && x2>=0 && x2<H) out[cnt][x2][i]=ww2;
                    }
                }else{
                    int y1 = clampT((int)floor(w.c2.y/w_gap - 0.5f), 0, W-1);
                    int y2 = clampT(y1+1, 0, W-1);
                    float ww1 = y2 - (w.c2.y/w_gap - 0.5f);
                    float ww2 = (w.c2.y/w_gap - 0.5f) - y1;
                    int x1 = clampT((int)floor(w.c1.x/h_gap - 0.5f), 0, H-1);
                    int x2 = clampT((int)ceil (w.c2.x/h_gap - 0.5f),  0, H-1);
                    for (int i=x1;i<=x2;++i){
                        if (i>=0 && i<H && y1>=0 && y1<W) out[cnt][i][y1]=ww1;
                        if (i>=0 && i<H && y2>=0 && y2<W) out[cnt][i][y2]=ww2;
                    }
                }
            }
            ++cnt;
        }

        for (int m=0; m<M-2; ++m){
            for (int h=0;h<H;++h)
                for (int w=0; w<W; ++w)
                    out[cnt][h][w] = out[m][h][w] * out[m+1][h][w];
            ++cnt;
        }

        return out;
    }

    boost::multi_array<float,3> run(const string& netlist){
        read_data(netlist);
        merge_wires();
        merge_nodes();
        build_partitions();
        compute_currents();
        compute_ir_drop();

        auto cols = collect_ir_columns();
        auto ir   = build_ir_map(cols);
        auto dist = build_distance();
        auto mask = build_res_mask();

        int C1=ir.shape()[0], C2=dist.shape()[0], C3=mask.shape()[0];
        boost::multi_array<float,3> all(boost::extents[C1+C2+C3][H][W]);
        auto copy_plane=[&](auto& src,int sc, auto& dst,int dc){
            for (int h=0;h<H;++h) for (int w=0;w<W;++w) dst[dc][h][w]=src[sc][h][w];
        };
        int dc=0;
        for(int c=0;c<C1;++c,++dc) copy_plane(ir,c,all,dc);
        for(int c=0;c<C2;++c,++dc) copy_plane(dist,c,all,dc);
        for(int c=0;c<C3;++c,++dc) copy_plane(mask,c,all,dc);
        return all;
    }
};

// ---------------- CSV & stats ----------------
static void ensure_dir(const string& d){
    std::error_code ec;
    std::filesystem::create_directories(d, ec);
}
static void write_csv_plane(const string& path, const boost::multi_array<float,3>& a, int c){
    ofstream out(path);
    int H=a.shape()[1], W=a.shape()[2];
    for (int h=0; h<H; ++h){
        for (int w=0; w<W; ++w){
            out << a[c][h][w];
            if (w+1<W) out << ',';
        }
        out << '\n';
    }
}
static tuple<float,float,double> plane_stats(const boost::multi_array<float,3>& a, int c){
    int H=a.shape()[1], W=a.shape()[2];
    float mn=std::numeric_limits<float>::infinity();
    float mx=-mn; double sum=0.0;
    for (int h=0;h<H;++h) for (int w=0;w<W;++w){
        float v=a[c][h][w];
        mn=min(mn,v); mx=max(mx,v); sum += (double)v;
    }
    return {mn,mx,sum};
}

// ---------------- orientation pre-scan (auto) ----------------
struct ScanResult {
    vector<string> metals;
    vector<int>    orientation; // deduced
    map<string, pair<long,long>> counts; // metal -> (vertical,horizontal)
};

static ScanResult scan_netlist_for_orientation(const string& path){
    namespace io = boost::iostreams;  // ✅ 네임스페이스 별칭
    set<string> metal_set;
    map<string, pair<long,long>> cnt; // (v,h)

    auto handle_line = [&](const string& line){
        auto toks = MultiSplitStringV2(line, " _");
        if (toks.empty()) return;
        char head = toks[0].empty()?'\0':toks[0][0];
        if (head=='R'){
            if (toks.size()<10) return;
            string m1 = norm_layer(string(toks[2]));
            string m2 = norm_layer(string(toks[6]));
            metal_set.insert(m1); metal_set.insert(m2);
            if (m1==m2){
                int x1 = atoi(string(toks[3]).c_str());
                int y1 = atoi(string(toks[4]).c_str());
                int x2 = atoi(string(toks[7]).c_str());
                int y2 = atoi(string(toks[8]).c_str());
                auto& c = cnt[m1];
                if (x1!=x2 && y1==y2) ++c.first;     // vertical
                else if (y1!=y2 && x1==x2) ++c.second; // horizontal
            }
        }else if (head=='I' || head=='V'){
            if ((head=='I' && toks.size()<7) || (head=='V' && toks.size()<5)) return;
            string m1 = norm_layer(string(toks[2]));
            metal_set.insert(m1);
        }
    };

    if (ends_with(path,".gz")){
        io::filtering_istream in; in.push(io::gzip_decompressor()); in.push(io::file_source(path));
        if (!in.good()) throw runtime_error("open gz failed in scan: "+path);
        string line; while (getline(in,line)) if (!line.empty()) handle_line(line);
    }else{
        ifstream in(path); if (!in) throw runtime_error("open failed in scan: "+path);
        string line; while (getline(in,line)) if (!line.empty()) handle_line(line);
    }

    vector<string> metals(metal_set.begin(), metal_set.end());
    sort(metals.begin(), metals.end(), [](const string& a, const string& b){
        int na=layer_num(a), nb=layer_num(b);
        if (na != nb) return na < nb;
        return a < b;
    });

    vector<int> ori; ori.reserve(metals.size());
    for (auto& m : metals){
        auto it = cnt.find(m);
        long v = (it==cnt.end()?0:it->second.first);
        long h = (it==cnt.end()?0:it->second.second);
        if (v>h) ori.push_back(VERTICAL);
        else if (h>v) ori.push_back(HORIZONTAL);
        else          ori.push_back(VERTICAL); // tie/default
    }
    return {metals, ori, cnt};
}

static vector<string> split_csv(const string& s){
    vector<string> out;
    string cur; std::stringstream ss(s);
    while (getline(ss, cur, ',')) if(!cur.empty()) out.push_back(cur);
    return out;
}
static vector<int> parse_ori_list(const string& s){
    vector<int> out;
    auto toks = split_csv(s);
    for (auto t : toks){
        string u = tolower_copy(t);
        if (u=="v" || u=="vertical" || u=="1") out.push_back(VERTICAL);
        else if (u=="h" || u=="horizontal" || u=="2") out.push_back(HORIZONTAL);
        else throw runtime_error("invalid orientation token: "+t);
    }
    return out;
}
static void print_ori_table(const vector<string>& metals, const vector<int>& ori){
    cerr << "Orientation:\n";
    for (size_t i=0;i<metals.size();++i){
        cerr << "  " << metals[i] << " : " << (ori[i]==VERTICAL?"VERTICAL":"HORIZONTAL") << "\n";
    }
}

// ---------------- main ----------------
int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc<4){
        cerr <<
        "Usage: irdump <netlist.sp[.gz]> <H> <W> [options]\n\n"
        "Options:\n"
        "  --preset {sky130|asap7|nangate45|custom}  (default: sky130)\n"
        "  --auto-ori      Auto-detect metals & orientation from file (overrides preset metals/orientation)\n"
        "  --metals L1,L2,...   (custom preset only; names like m1, M3, met5 ok)\n"
        "  --ori    O1,O2,...   (custom preset only; V/H or 1/2)\n"
        "  --out-dir DIR        Dump per-channel CSVs to DIR\n"
        "  --print-ori         Print metals/orientation used\n";
        return 1;
    }

    string net = argv[1];
    int H = stoi(argv[2]);
    int W = stoi(argv[3]);

    string preset = "sky130";
    bool auto_ori = false;
    string out_dir;
    bool print_ori = false;
    string metals_csv, ori_csv;

    for (int i=4;i<argc;++i){
        string a=argv[i];
        if (a=="--preset" && i+1<argc){ preset = tolower_copy(argv[++i]); }
        else if (a=="--auto-ori"){ auto_ori = true; }
        else if (a=="--out-dir" && i+1<argc){ out_dir = argv[++i]; }
        else if (a=="--print-ori"){ print_ori = true; }
        else if (a=="--metals" && i+1<argc){ metals_csv = argv[++i]; }
        else if (a=="--ori" && i+1<argc){ ori_csv = argv[++i]; }
        else {
            cerr << "Unknown/invalid arg: " << a << "\n";
            return 1;
        }
    }

    // Decide metals/orientations
    vector<string> metals;
    vector<int> ori;

    if (auto_ori){
        auto sc = scan_netlist_for_orientation(net);
        metals = sc.metals;
        ori    = sc.orientation;
        if (metals.empty()) { cerr << "Auto-orientation found no metals.\n"; return 2; }
    }else if (preset=="sky130"){
        metals = {"met1","met4","met5"};
        ori    = {VERTICAL, HORIZONTAL, VERTICAL};
    }else if (preset=="asap7"){
        // Common in many ASAP7 BeGAN benches (adjust if needed)
        metals = {"m5","m8","m9","m10"};        // m*, M*, met* 모두 허용됨
        for (auto& s: metals) s = norm_layer(s);
        ori    = {HORIZONTAL, VERTICAL, HORIZONTAL, VERTICAL};
    }else if (preset=="nangate45"){
        // Frequently seen selection; override with --auto-ori if mismatched
        metals = {"m1","m4","m7","m8","m9"};
        for (auto& s: metals) s = norm_layer(s);
        ori    = {VERTICAL, HORIZONTAL, VERTICAL, HORIZONTAL, VERTICAL};
    }else if (preset=="custom"){
        if (metals_csv.empty() || ori_csv.empty()){
            cerr << "custom preset requires --metals and --ori\n";
            return 1;
        }
        metals = split_csv(metals_csv);
        for (auto& s: metals) s = norm_layer(s);
        ori    = parse_ori_list(ori_csv);
        if (metals.size()!=ori.size()){
            cerr << "metals and ori length mismatch\n"; return 1;
        }
        // sort by layer number but keep orientation aligned
        vector<int> idx(metals.size()); iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&](int a,int b){
            int na=layer_num(metals[a]), nb=layer_num(metals[b]);
            if (na != nb) return na < nb;
            return metals[a] < metals[b];
        });
        vector<string> m2; vector<int> o2; m2.reserve(idx.size()); o2.reserve(idx.size());
        for (int id: idx){ m2.push_back(metals[id]); o2.push_back(ori[id]); }
        metals.swap(m2); ori.swap(o2);
    }else{
        cerr << "Unknown preset: " << preset << "\n"; return 1;
    }

    if (print_ori) print_ori_table(metals, ori);

    try{
        Circuit cir;
        cir.set_orientation(metals, ori);
        cir.set_size(H,W);

        auto all = cir.run(net);
        int C = all.shape()[0];

        cout << "Channels: " << C << "  (H="<<H<<", W="<<W<<")\n";
        for (int c=0;c<C;++c){
            auto [mn,mx,sum] = plane_stats(all,c);
            cout << "  ch" << c << "  min=" << mn << "  max=" << mx << "  sum=" << sum << "\n";
        }

        if (!out_dir.empty()){
            ensure_dir(out_dir);
            for (int c=0;c<C;++c){
                std::ostringstream oss;
                oss << out_dir << "/channel_" << c << ".csv";
                write_csv_plane(oss.str(), all, c);
            }
            cout << "Saved CSVs to: " << out_dir << "\n";
        }
    }catch(const std::exception& e){
        cerr << "Error: " << e.what() << "\n";
        return 2;
    }
    return 0;
}
