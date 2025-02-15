#include <AMReX.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

namespace {
    std::vector<std::string> get_node_list (std::string const& job_id)
    {
        std::string cmd("sacct --parsable2 --format=NNodes,NodeList -X -j ");
        cmd.append(job_id);

        std::string r;
        if (FILE* ps = popen(cmd.c_str(), "r")) {
            char print_buff[512];
            while (fgets(print_buff, sizeof(print_buff), ps)) {
                r += print_buff;
            }
            pclose(ps);
        }

        auto ferror = [&] () -> std::vector<std::string> {
            std::cout << "Unexpected result from " << cmd << ": " << r << "\n";
            amrex::Abort();
            return {};
        };

        auto lines = amrex::split(r, "\n");
        if (lines.size() != 2) { return ferror(); }

        auto words = amrex::split(lines[1],"|");
        if (words.size() != 2) { return ferror(); }

        int nnodes;
        {
            std::stringstream ss(words[0]);
            ss >> nnodes;
        }
        auto words2 = amrex::split(words[1], "[]");
        if (words2.size() < 2) { return ferror(); }

        auto nodes = amrex::split(words2[1],",-");
        if (nnodes != nodes.size()) { return ferror(); }
                                  
        return nodes;
    }
}

int main(int argc, char* argv[])
{
    {
        ParmParse pp("amrex");
        pp.add("verbose", 0);
    }
    amrex::Initialize(argc,argv);
    {
        std::set<std::string> good_jobs;
        std::set<std::string> bad_jobs;
        {
            std::vector<std::string> good_jobs_v;
            std::vector<std::string> bad_jobs_v;
            ParmParse pp;
            pp.getarr("good_jobs", good_jobs_v);
            pp.getarr("bad_jobs", bad_jobs_v);
            good_jobs.insert(good_jobs_v.begin(), good_jobs_v.end());
            bad_jobs.insert(bad_jobs_v.begin(), bad_jobs_v.end());
        }

        std::vector<std::string> good_nodes;
        std::cout << "\n";
        for (auto const& job_id : good_jobs) {
            std::cout << "Processing good job: " << job_id << ", ";
            auto nodes = get_node_list(job_id);
            std::cout << "Number of nodes: " << nodes.size() << "\n";
            good_nodes.insert(good_nodes.end(), nodes.begin(), nodes.end());
        }
        std::sort(good_nodes.begin(), good_nodes.end());
        auto git = std::unique(good_nodes.begin(), good_nodes.end());
        good_nodes.erase(git, good_nodes.end());

        std::cout << "\nNumber of Good Nodes: " << good_nodes.size() << "\n\n";

        std::set<std::string> bad_nodes;
        std::map<std::string,int> maybe_bad_nodes;
        for (auto const& job_id: bad_jobs) {
            std::cout << "Processing bad job: " << job_id << ", ";
            auto nodes = get_node_list(job_id);
            std::cout << "Number of nodes: " << nodes.size() << "\n";

            std::vector<std::string> maybe;
            for (auto const& node : nodes) {
                auto found = std::find(good_nodes.cbegin(), good_nodes.cend(), node);
                if (found == good_nodes.cend()) {
                    maybe.push_back(node);
                }
            }
            if (maybe.empty()) {
                std::cout << "WARNING: How did this happend?! "
                    "Some of the so-called good nodes may not be good after all.\n";
            } else if (maybe.size() == 1) {
                bad_nodes.insert(maybe[0]);
            } else {
                for (auto const& node : maybe) {
                    ++maybe_bad_nodes[node];
                }
            }
        }

        if (bad_nodes.size() == 1) {
            std::cout << "\n" << bad_nodes.size() << " Bad Node: ";
        } else {
            std::cout << "\n" << bad_nodes.size() << " Bad Nodes: ";
        }
        bool first = true;
        for (auto const& node : bad_nodes) {
            if (first) {
                first = false;
            } else {
                std::cout << ",";
            }
            std::cout << node;
        }
        std::cout << "\n";

        std::cout << "\nNumber of Maybe Bad Nodes: " << maybe_bad_nodes.size() << "\n";
        {
            std::map<int,std::vector<std::string>> maybe_bad_nodes_2;
            for (auto const& [node,count] : maybe_bad_nodes) {
                if (bad_nodes.find(node) == bad_nodes.end()) {
                    maybe_bad_nodes_2[count].push_back(node);
                }
            }
            int count = 0;
            for (auto it = maybe_bad_nodes_2.rbegin();
                 it != maybe_bad_nodes_2.rend() && count < 2;
                 ++it, ++count)
            {
                std::cout << "    " << it->second.size() << " nodes appeared to be bad "
                          << it->first << " times: ";
                for (std::size_t i = 0; i < it->second.size(); ++i) {
                    if (i == 0) {
                        std::cout << it->second[i];
                    } else {
                        std::cout << "," << it->second[i];
                    }
                }
                std::cout << "\n\n";
            }
        }
    }
    amrex::Finalize();
}
