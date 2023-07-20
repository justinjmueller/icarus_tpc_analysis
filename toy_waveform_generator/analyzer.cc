#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <ctime>

#include "TFile.h"
#include "TNtuple.h"

double median(std::vector<double>& v)
{
    size_t n = v.size() / 2;
    nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}

double calc_pedestal(std::vector<double>& w, unsigned short p=100)
{
    std::vector<double> sorted(w);
    std::sort(sorted.begin(), sorted.end(), [](const auto& l, const auto& r){return std::fabs(l) < std::fabs(r);});
    return std::accumulate(w.begin(), w.end()-p, 0.0) / (w.size()-p);
}

double calc_rms(std::vector<double>& w, size_t l=0)
{
    return std::sqrt(std::inner_product(w.begin(), l==0 ? w.end() : w.begin() + l, w.begin(), 0.0) / double(l==0 ? w.size() : l));
}

void extract_coherent(std::vector<std::vector<double> >::const_iterator first,
                      std::vector<std::vector<double> >::const_iterator last,
                      std::vector<double>& coherent)
{
    coherent.resize(4096);
    std::vector<double> current_tick(size_t(last-first));
    for(size_t t(0); t < first->size(); ++t)
    {
        for(auto it(first); it != last; ++it)
            current_tick.at(size_t(it-first)) = it->at(t);
        coherent.at(t) = median(current_tick);
    }
}

std::vector<std::string> split(const std::string& line, char delimiter) 
{
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

void load_event(const std::string file_name, std::vector<std::vector<double> >& waveforms)
{
    std::ifstream file(file_name);
    std::string line;
    while (std::getline(file, line))
    {
        std::vector<std::string> tokens = split(line, ',');
        std::vector<double> row;
        for(const auto& token : tokens)
            row.push_back(std::stod(token));
        waveforms.push_back(row);
    }
}

void process_event(TNtuple *tuple, const size_t event)
{
    std::vector<std::vector<double> > raw_waveforms;
    std::stringstream name;
    name << std::setfill('0') << std::setw(4) << event;
    load_event("/Volumes/MUSB/noise_model/coh_event" + name.str(), raw_waveforms);
    //load_event("/Volumes/MUSB/waveforms/run9394_frag4108_evt" + std::to_string(event), raw_waveforms);

    std::vector<double> pedestals;
    std::vector<std::vector<double> > coh_waveforms(raw_waveforms.size() / 32);
    std::vector<std::vector<double> > int_waveforms(raw_waveforms.size());
    float raw_rms, coh_rms, int_rms, range;
    for(size_t c(0); c < raw_waveforms.size(); ++c)
    {
        pedestals.push_back(calc_pedestal(raw_waveforms[c]));
        std::transform(raw_waveforms[c].begin(), raw_waveforms[c].end(), raw_waveforms[c].begin(),
                       std::bind(std::minus<double>(), std::placeholders::_1, pedestals[c]));
    }
    for(size_t g(0); g < raw_waveforms.size() / 32; ++g)
    {
        extract_coherent(raw_waveforms.begin()+2*int(g/2)*32, //g*32,
                         raw_waveforms.begin()+(2*int(g/2)+2)*32, //(g+1)*32,
                         coh_waveforms.at(g));
        coh_rms = calc_rms(coh_waveforms.at(g));

        for(uint16_t c(0); c < 32; ++c)
        {
            int_waveforms.at(g*32+c).resize(4096);
            std::transform(raw_waveforms.at(g*32+c).begin(),
                           raw_waveforms.at(g*32+c).end(),
                           coh_waveforms.at(g).begin(),
                           int_waveforms.at(g*32+c).begin(),
                           std::minus<double>());

            raw_rms = calc_rms(raw_waveforms.at(g*32+c));
            int_rms = calc_rms(int_waveforms.at(g*32+c));
            range = (*std::max_element(int_waveforms.at(g*32+c).begin(),
                                       int_waveforms.at(g*32+c).end())
                     - *std::min_element(int_waveforms.at(g*32+c).begin(),
                                         int_waveforms.at(g*32+c).end()));
            tuple->Fill(0, event, std::time(nullptr), 4108, int(g/2), int(g/2), (g%2)*32+c,
                        pedestals.at((g/2)*64+(g%2)*32+c), raw_rms, int_rms, coh_rms, range);
        }
    }
}

int main()
{
    TFile output("run3.root", "recreate");
    TNtuple *tuple = new TNtuple("tpcnoise", "TPCNoiseToyMC", "run:event:time:frag:board:slot_id:ch:ped:rms:intrms:cohrms:range");

    for(size_t n(0); n < 100; ++n)
        process_event(tuple, n);

    tuple->Write();
    output.Close();

    return 0;
}