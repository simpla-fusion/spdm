#include <string>

class Entry
{
    Entry get(std::string const&){};
};

int task_api(Entry& in_entry, Entry& out_entry)
{
    auto n = in_entry.get("equilibrium.profile_1d.density").get<array>();
};

int eq_solver(Entry& in_entry, Entry& out_entry)
{
    auto n = in_entry.get("equilibrium.profile_1d.density").get<array>();
};

class Equilibrium;
class CoreTransport;

int eq_solver(Equilibrium& eq, CoreTransport& trans, Entry& out_entry)
{
    eq = in_etry.get("equilibrium");
}
