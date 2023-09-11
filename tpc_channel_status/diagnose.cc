#include <iostream>
#include <fstream>
#include <vector>
#include <map>

#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TParameter.h"
#include "sqlite3.h"

#define EVENT_MAX 1000
#define NOISE_THRESHOLD_LOW 2.00
#define NOISE_THRESHOLD_HIGH 7.00
#define NOISE_NORMED_THRESHOLD_LOW 0.005
#define NOISE_NORMED_THRESHOLD_HIGH 0.015
#define SIGNAL_THRESHOLD_HEIGHT 20

static int callback_flatcable(void *output, int count, char **data, char **columns)
{
    std::map<std::string, double> *cable_map = static_cast<std::map<std::string, double>* >(output);
    cable_map->insert(std::make_pair(std::string(data[0]), std::stof(data[1])));
    return 0;
}

static int callback_physicalwire(void *output, int count, char **data, char **columns)
{
    std::map<uint16_t, double> *cable_map = static_cast<std::map<uint16_t, double>* >(output);
    cable_map->insert(std::make_pair(std::stoi(data[0]), std::stof(data[1])));
    return 0;
}

static int callback_daqchannel(void *output, int count, char **data, char **columns)
{
    std::map<uint16_t, std::string> *cable_map = static_cast<std::map<uint16_t, std::string>* >(output);
    cable_map->insert(std::make_pair(std::stoi(data[0]), std::string(data[1])));
    return 0;
}

void build_map(std::map<uint16_t, double> &capacitance, std::map<uint16_t, double> &cable_length, std::map<uint16_t, double> &wire_length)
{
    sqlite3* db;
    if(sqlite3_open("/Users/mueller/Projects/channel_status/db/icarus_channels_dev.db", &db) == SQLITE_OK)
    {
        std::map<std::string, double> cable_map;
        sqlite3_exec(db, "SELECT cable_number, capacitance FROM flatcables", callback_flatcable, &cable_map, NULL);

        std::map<uint16_t, double> wire_map;
        sqlite3_exec(db, "SELECT channel_id, capacitance FROM physicalwires", callback_physicalwire, &wire_map, NULL);

        std::map<uint16_t, std::string> channel_map;
        sqlite3_exec(db, "SELECT channel_id, cable_number FROM channelinfo", callback_daqchannel, &channel_map, NULL);
        sqlite3_close(db);

        for(const std::pair<uint16_t, std::string> p : channel_map)
        {
            if(cable_map.find(p.second) != cable_map.end() && wire_map.find(p.first) != wire_map.end())
            {
                capacitance.insert(std::make_pair(p.first, cable_map[p.second] + wire_map[p.first] + 30.));
                cable_length.insert(std::make_pair(p.first, cable_map[p.second] / 47.0));
                if(p.first % 13824 < 2304 || p.first % 13824 > 8064)
                    wire_length.insert(std::make_pair(p.first, wire_map[p.first] / 20.0));
                else
                    wire_length.insert(std::make_pair(p.first, wire_map[p.first] / 21.0));
            }
            else
            {
                capacitance.insert(std::make_pair(p.first, 30.0));
                cable_length.insert(std::make_pair(p.first, 0.0));
                wire_length.insert(std::make_pair(p.first, 0.0));
            }
        }
    }
}

int main(int argc, char **argv)
{
    std::map<uint16_t, double> capacitance, cable_lengths, wire_lengths;
    build_map(capacitance, cable_lengths, wire_lengths);

    TH1D *mhit_height_plane0 = new TH1D("mhit_height_plane0", "mhit_height_plane0", 100, 0, 100);
    TH1D *mhit_height_plane1 = new TH1D("mhit_height_plane1", "mhit_height_plane1", 100, 0, 100);
    TH1D *mhit_height_plane2 = new TH1D("mhit_height_plane2", "mhit_height_plane2", 100, 0, 100);
    TH1D *capacitance_plane0 = new TH1D("capacitance_plane0", "capacitance_plane0", 75, 0, 750);
    TH1D *capacitance_plane1 = new TH1D("capacitance_plane1", "capacitance_plane1", 75, 0, 750);
    TH1D *capacitance_plane2 = new TH1D("capacitance_plane2", "capacitance_plane2", 75, 0, 750);

    TH1D *raw_rms_full_plane0 = new TH1D("raw_rms_full_plane0", "raw_rms_full_plane0", 250, 0, 10);
    TH1D *raw_rms_full_plane1 = new TH1D("raw_rms_full_plane1", "raw_rms_full_plane1", 250, 0, 10);
    TH1D *raw_rms_full_plane2 = new TH1D("raw_rms_full_plane2", "raw_rms_full_plane2", 250, 0, 10);
    TH1D *int_rms_full_plane0 = new TH1D("int_rms_full_plane0", "int_rms_full_plane0", 250, 0, 10);
    TH1D *int_rms_full_plane1 = new TH1D("int_rms_full_plane1", "int_rms_full_plane1", 250, 0, 10);
    TH1D *int_rms_full_plane2 = new TH1D("int_rms_full_plane2", "int_rms_full_plane2", 250, 0, 10); 

    TH1D *raw_rms_sigf_plane0 = new TH1D("raw_rms_sigf_plane0", "raw_rms_sigf_plane0", 250, 0, 10);
    TH1D *raw_rms_sigf_plane1 = new TH1D("raw_rms_sigf_plane1", "raw_rms_sigf_plane1", 250, 0, 10);
    TH1D *raw_rms_sigf_plane2 = new TH1D("raw_rms_sigf_plane2", "raw_rms_sigf_plane2", 250, 0, 10);
    TH1D *int_rms_sigf_plane0 = new TH1D("int_rms_sigf_plane0", "int_rms_sigf_plane0", 250, 0, 10);
    TH1D *int_rms_sigf_plane1 = new TH1D("int_rms_sigf_plane1", "int_rms_sigf_plane1", 250, 0, 10);
    TH1D *int_rms_sigf_plane2 = new TH1D("int_rms_sigf_plane2", "int_rms_sigf_plane2", 250, 0, 10); 
    
    TH1D *raw_rms_norm_plane0 = new TH1D("raw_rms_norm_plane0", "raw_rms_norm_plane0", 250, 0, 0.04);
    TH1D *raw_rms_norm_plane1 = new TH1D("raw_rms_norm_plane1", "raw_rms_norm_plane1", 250, 0, 0.04);
    TH1D *raw_rms_norm_plane2 = new TH1D("raw_rms_norm_plane2", "raw_rms_norm_plane2", 250, 0, 0.04);
    TH1D *int_rms_norm_plane0 = new TH1D("int_rms_norm_plane0", "int_rms_norm_plane0", 250, 0, 0.04);
    TH1D *int_rms_norm_plane1 = new TH1D("int_rms_norm_plane1", "int_rms_norm_plane1", 250, 0, 0.04);
    TH1D *int_rms_norm_plane2 = new TH1D("int_rms_norm_plane2", "int_rms_norm_plane2", 250, 0, 0.04);

    TH1D *occupancy_plane0 = new TH1D("occupancy_plane0", "occupancy_plane0", 100, 0, 1);
    TH1D *occupancy_plane1 = new TH1D("occupancy_plane1", "occupancy_plane1", 100, 0, 1);
    TH1D *occupancy_plane2 = new TH1D("occupancy_plane2", "occupancy_plane2", 100, 0, 1);
    TH1D *occupancy_norm_plane0 = new TH1D("occupancy_norm_plane0", "occupancy_norm_plane0", 100, 0, 20);
    TH1D *occupancy_norm_plane1 = new TH1D("occupancy_norm_plane1", "occupancy_norm_plane1", 100, 0, 20);
    TH1D *occupancy_norm_plane2 = new TH1D("occupancy_norm_plane2", "occupancy_norm_plane2", 100, 0, 20);

    TH2D *capacitance_intrms_norm = new TH2D("capacitance_intrms_norm", "capacitance_intrms_norm", 250, 0, 0.04, 75, 0, 750);

    TFile *input = new TFile(argv[1], "read");
    TTree *tree = static_cast<TTree*>(input->Get("tpcnoise"));
    uint16_t channel_id;
    double raw_rms, int_rms;
    double mhit_height;

    tree->SetBranchAddress("channel_id", &channel_id);
    tree->SetBranchAddress("raw_rms", &raw_rms);
    tree->SetBranchAddress("int_rms", &int_rms);
    tree->SetBranchAddress("mhit_height", &mhit_height);

    std::map<uint16_t, uint32_t> rms_low, rms_high;
    std::map<uint16_t, uint32_t> rms_normed_low, rms_normed_high;
    std::map<uint16_t, uint32_t> mhit_signal;
    std::map<uint16_t, double> intrms_sum, intrms_squaresum;

    uint32_t nevents(tree->GetEntries() / 54784);
    if(nevents > EVENT_MAX) nevents = EVENT_MAX;
    for(size_t n(0); n < tree->GetEntries() && n < nevents * 54784; ++n)
    {
        tree->GetEntry(n);
        if(mhit_height < SIGNAL_THRESHOLD_HEIGHT)
            capacitance_intrms_norm->Fill(int_rms / capacitance[channel_id], capacitance[channel_id]);
        if(channel_id % 13824 < 2304)
        {
            if(mhit_height != 0) mhit_height_plane0->Fill(mhit_height);
            else mhit_height_plane0->Fill(-1);
            raw_rms_full_plane0->Fill(raw_rms);
            int_rms_full_plane0->Fill(int_rms);
            capacitance_plane0->Fill(capacitance[channel_id]);
            if(mhit_height < SIGNAL_THRESHOLD_HEIGHT)
            {
                raw_rms_sigf_plane0->Fill(raw_rms);
                int_rms_sigf_plane0->Fill(int_rms);
                raw_rms_norm_plane0->Fill(raw_rms / capacitance[channel_id]);
                int_rms_norm_plane0->Fill(int_rms / capacitance[channel_id]);
            }
        }
        else if(channel_id % 13824 < 8064)
        {
            if(mhit_height != 0) mhit_height_plane1->Fill(mhit_height);
            else mhit_height_plane1->Fill(-1);
            raw_rms_full_plane1->Fill(raw_rms);
            int_rms_full_plane1->Fill(int_rms);
            capacitance_plane1->Fill(capacitance[channel_id]);
            if(mhit_height < SIGNAL_THRESHOLD_HEIGHT)
            {
                raw_rms_sigf_plane1->Fill(raw_rms);
                int_rms_sigf_plane1->Fill(int_rms);
                raw_rms_norm_plane1->Fill(raw_rms / capacitance[channel_id]);
                int_rms_norm_plane1->Fill(int_rms / capacitance[channel_id]);
            }
        }
        else
        {
            if(mhit_height != 0) mhit_height_plane2->Fill(mhit_height);
            else mhit_height_plane2->Fill(-1);
            raw_rms_full_plane2->Fill(raw_rms);
            int_rms_full_plane2->Fill(int_rms);
            capacitance_plane2->Fill(capacitance[channel_id]);
            if(mhit_height < SIGNAL_THRESHOLD_HEIGHT)
            {
                raw_rms_sigf_plane2->Fill(raw_rms);
                int_rms_sigf_plane2->Fill(int_rms);
                raw_rms_norm_plane2->Fill(raw_rms / capacitance[channel_id]);
                int_rms_norm_plane2->Fill(int_rms / capacitance[channel_id]);
            }
        }
        if(rms_low.find(channel_id) == rms_low.end())
        {
            rms_low.insert(std::make_pair(channel_id, 0));
            rms_high.insert(std::make_pair(channel_id, 0));
            mhit_signal.insert(std::make_pair(channel_id, 0));
            intrms_sum.insert(std::make_pair(channel_id, 0.0));
            intrms_squaresum.insert(std::make_pair(channel_id, 0.0));
        }        
        if(mhit_height < SIGNAL_THRESHOLD_HEIGHT)
        {
            if(raw_rms < NOISE_THRESHOLD_LOW) ++rms_low[channel_id];
            if(raw_rms > NOISE_THRESHOLD_HIGH) ++rms_high[channel_id];
            if(int_rms / capacitance[channel_id] < NOISE_NORMED_THRESHOLD_LOW) ++rms_normed_low[channel_id];
            if(int_rms / capacitance[channel_id] > NOISE_NORMED_THRESHOLD_HIGH) ++rms_normed_high[channel_id];
        }
        else ++mhit_signal[channel_id];
        intrms_sum[channel_id] += int_rms;
        intrms_squaresum[channel_id] += int_rms*int_rms;

        if(channel_id == 1795)
            std::cerr << "Channel RMS: " << raw_rms << std::endl;
    }

    input->Close();
    delete input;

    TFile *output = new TFile(argv[2], "recreate");

    TParameter<double> event_count("event_count", nevents);
    event_count.Write();

    output->WriteObject(mhit_height_plane0, "mhit_height_plane0");
    output->WriteObject(mhit_height_plane1, "mhit_height_plane1");
    output->WriteObject(mhit_height_plane2, "mhit_height_plane2");
    output->WriteObject(capacitance_plane0, "capacitance_plane0");
    output->WriteObject(capacitance_plane1, "capacitance_plane1");
    output->WriteObject(capacitance_plane2, "capacitance_plane2");
    output->WriteObject(raw_rms_full_plane0, "raw_rms_full_plane0");
    output->WriteObject(raw_rms_full_plane1, "raw_rms_full_plane1");
    output->WriteObject(raw_rms_full_plane2, "raw_rms_full_plane2");
    output->WriteObject(int_rms_full_plane0, "int_rms_full_plane0");
    output->WriteObject(int_rms_full_plane1, "int_rms_full_plane1");
    output->WriteObject(int_rms_full_plane2, "int_rms_full_plane2");
    output->WriteObject(raw_rms_sigf_plane0, "raw_rms_sigf_plane0");
    output->WriteObject(raw_rms_sigf_plane1, "raw_rms_sigf_plane1");
    output->WriteObject(raw_rms_sigf_plane2, "raw_rms_sigf_plane2");
    output->WriteObject(int_rms_sigf_plane0, "int_rms_sigf_plane0");
    output->WriteObject(int_rms_sigf_plane1, "int_rms_sigf_plane1");
    output->WriteObject(int_rms_sigf_plane2, "int_rms_sigf_plane2");
    output->WriteObject(raw_rms_norm_plane0, "raw_rms_norm_plane0");
    output->WriteObject(raw_rms_norm_plane1, "raw_rms_norm_plane1");
    output->WriteObject(raw_rms_norm_plane2, "raw_rms_norm_plane2");
    output->WriteObject(int_rms_norm_plane0, "int_rms_norm_plane0");
    output->WriteObject(int_rms_norm_plane1, "int_rms_norm_plane1");
    output->WriteObject(int_rms_norm_plane2, "int_rms_norm_plane2");
    output->WriteObject(capacitance_intrms_norm, "capacitance_intrms_norm");
    tree = new TTree("channels", "channels");

    uint16_t channel_id_output;
    double low_noise, high_noise, low_noise_normed, high_noise_normed;
    double occupancy, occupancy_normed;
    double mean_intrms, intrms_normed, cable_length, wire_length, std_intrms;
    tree->Branch("channel_id", &channel_id_output);
    tree->Branch("low_noise", &low_noise);
    tree->Branch("high_noise", &high_noise);
    tree->Branch("low_noise_normed", &low_noise_normed);
    tree->Branch("high_noise_normed", &high_noise_normed);
    tree->Branch("occupancy", &occupancy);
    tree->Branch("occupancy_normed", &occupancy_normed);
    tree->Branch("mean_intrms", &mean_intrms);
    tree->Branch("intrms_normed", &intrms_normed);
    tree->Branch("cable_length", &cable_length);
    tree->Branch("wire_length", &wire_length);
    tree->Branch("std_intrms", &std_intrms);

    for(const std::pair<uint16_t, uint32_t>& p : rms_low)
    {
        channel_id_output = p.first;
        low_noise = (double)rms_low[p.first] / nevents;
        high_noise = (double)rms_high[p.first] / nevents;
        low_noise_normed = (double)rms_normed_low[p.first] / nevents;
        high_noise_normed = (double)rms_normed_high[p.first] / nevents;
        occupancy = (double)mhit_signal[p.first] / nevents;
        occupancy_normed = 100*(occupancy / wire_lengths[p.first]);
        mean_intrms = (double)intrms_sum[p.first] / nevents;
        intrms_normed = mean_intrms / capacitance[p.first];
        cable_length = cable_lengths[p.first];
        wire_length = wire_lengths[p.first];
        std_intrms = std::sqrt(((double)intrms_squaresum[p.first] / nevents) - mean_intrms*mean_intrms);
        tree->Fill();

        if(channel_id_output % 13824 < 2304)
        {
            occupancy_plane0->Fill(occupancy);
            occupancy_norm_plane0->Fill(occupancy_normed);
        }
        else if(channel_id_output % 13824 < 8064)
        {
            occupancy_plane1->Fill(occupancy);
            occupancy_norm_plane1->Fill(occupancy_normed);
        }
        else
        {
            occupancy_plane2->Fill(occupancy);
            occupancy_norm_plane2->Fill(occupancy_normed);
        }
    }
    tree->Write();
    output->WriteObject(occupancy_plane0, "occupancy_plane0");
    output->WriteObject(occupancy_plane1, "occupancy_plane1");
    output->WriteObject(occupancy_plane2, "occupancy_plane2");
    output->WriteObject(occupancy_norm_plane0, "occupancy_norm_plane0");
    output->WriteObject(occupancy_norm_plane1, "occupancy_norm_plane1");
    output->WriteObject(occupancy_norm_plane2, "occupancy_norm_plane2");
    output->Close();
    delete output;
}