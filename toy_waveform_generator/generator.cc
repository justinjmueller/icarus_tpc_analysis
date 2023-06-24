#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <random>

#define PI 3.14159265358979323846 

void generate_waveform(const size_t c, const std::vector<float>& data, std::vector<float>& waveform)
{
    waveform.resize(4096);
    float df(1.0 / (0.4*4096));
    float dt(0.4);
    std::vector<float> phase(2049);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> rng(0.0, 2*PI);
    for(size_t f(0); f < 2049; ++f)
        phase[f] = rng(gen);
    for(size_t t(0); t < 4096; ++t)
    {
        waveform[t] = 0;
        for(size_t f(0); f < 2049; ++f)
            waveform[t] += std::sqrt((0.4 / 4096) * data[c*2049+f]) * std::sin(df*f*t*dt+phase[f]);
    }
}

void load_input(const char* name, std::vector<float>& data)
{
    // Open input file.
    std::ifstream binary_file(std::string("/Users/mueller/Downloads/") + name, std::ios::binary);

    // Calculate length and resize vector.
    binary_file.seekg(0, std::ios::end);
    std::streampos file_size = binary_file.tellg();
    binary_file.seekg(0, std::ios::beg);
    data.resize(file_size / sizeof(float));
    
    // Read data into vector.
    binary_file.read(reinterpret_cast<char*>(data.data()), file_size);
}

float calc_rms(const std::vector<float>& waveform)
{
    return std::sqrt(std::inner_product(waveform.begin(), waveform.end(), waveform.begin(), 0) / 4096.0);
}

void write_event(const size_t n, const std::vector<std::vector<float> >& waveforms)
{
    std::ofstream output;
    std::stringstream name;
    name << std::setfill('0') << std::setw(4) << n;
    std::cout << "Writing to file: " << "/Volumes/MUSB/noise_model/event" + name.str() << std::endl;
    output.open("/Volumes/MUSB/noise_model/event" + name.str());
    for(size_t c(0); c < 576; ++c)
    {
        for(size_t t(0); t < 4096; ++t)
        {
            if(t < 4095) output << waveforms[c][t] << ",";
            else output << waveforms[c][t] << "\n";
        }
    }
    output.close();
}

int main()
{
    std::vector<float> raw_ffts;
    std::vector<std::vector<float> > waveforms(576);
    load_input("ifft.bin", raw_ffts);

    for(size_t n(0); n < 250; ++n)
    {
        for(size_t c(0); c < 576; ++c)
            generate_waveform(c, raw_ffts, waveforms[c]);
        write_event(n, waveforms);
    }
    return 0;
}