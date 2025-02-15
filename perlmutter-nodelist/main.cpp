#include <AMReX.H>
#include <AMReX_BLassert.H>
#include <AMReX_String.H>

#include <fstream>
#include <map>
#include <set>

using namespace amrex;

namespace {
    std::pair<std::string,std::vector<std::string>>
    get_node_list (std::string node_list_file)
    {
        std::ifstream ifs(node_list_file);
        if (! ifs.is_open()) { return {}; }

        std::string line;
        std::getline(ifs, line);
        auto words = amrex::split(line, "=");

        int nnodes;
        ifs >> nnodes;
        std::getline(ifs, line);

        std::getline(ifs, line);
        auto nodes = amrex::split(line, ",-");
        AMREX_ALWAYS_ASSERT(nnodes == nodes.size());

        return std::make_pair(amrex::trim(words[1]),nodes);
    }
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        std::cout << "\n";

        std::vector<std::string> good_nodes;
        for (int i = 0; i < 100; ++i) {
            auto [jobid, nodes] = get_node_list("good-nodelist-"+std::to_string(i));
            if (nodes.empty()) { break; }
            good_nodes.insert(good_nodes.end(), nodes.begin(), nodes.end());
        }
        std::sort(good_nodes.begin(), good_nodes.end());
        auto git = std::unique(good_nodes.begin(), good_nodes.end());
        good_nodes.erase(git, good_nodes.end());

        std::cout << "Number of Good Nodes: " << good_nodes.size() << "\n\n";

        std::vector<std::string> bad_nodes;
        std::map<std::string,int> maybe_bad_nodes;
        std::set<std::string> jobids;
        for (int i = 0; i < 100; ++i) {
            std::string fname("bad-nodelist-"+std::to_string(i));
            auto [jobid, nodes] = get_node_list(fname);
            if (nodes.empty()) { break; }
            if (jobids.count(jobid) == 1) {
                std::cout << fname << " for job " << jobid << " is redundant.\n";
            } else {
                jobids.insert(jobid);
                std::cout << "Processing " << fname << " for job " << jobid << "\n";
            }
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
                bad_nodes.push_back(maybe[0]);
            } else {
                for (auto const& node : maybe) {
                    ++maybe_bad_nodes[node];
                }
            }
        }
        std::sort(bad_nodes.begin(), bad_nodes.end());

        std::cout << "\nNumber of Bad Nodes: " << bad_nodes.size() << "\n";
        for (auto const& node : bad_nodes) {
            std::cout << "    " << node << "\n";
        }

        std::cout << "\nNumber of Maybe Bad Nodes: " << maybe_bad_nodes.size() << "\n";
        {
            std::map<int,std::vector<std::string>> maybe_bad_nodes_2;
            for (auto const& [node,count] : maybe_bad_nodes) {
                if (std::find(bad_nodes.begin(), bad_nodes.end(), node) == bad_nodes.end()) {
                    maybe_bad_nodes_2[count].push_back(node);
                }
            }
            for (auto it = maybe_bad_nodes_2.rbegin();
                 it != maybe_bad_nodes_2.rend(); ++it)
            {
                std::cout << "    " << it->second.size() << " nodes appeared to be bad "
                          << it->first << " times\n";
                for (auto const& node : it->second) {
                    std::cout << "        " << node << "\n";
                }
            }
        }

        std::cout << "\n";
    }
    amrex::Finalize();
}
