
#include <iostream>
#include <list>
#include <memory>
#include <sstream>

struct Foo
{
    Foo() : m_path_(new std::list<std::string>()) {}

    Foo(Foo const& other) : m_path_(new std::list<std::string>(*other.m_path_))
    {
        std::cout << "Copy construct " << str() << std::endl;
    };

    Foo(Foo&& other) : m_path_(other.m_path_.release())
    {
        std::cout << "Move construct " << m_path_.get() << std::endl;
    };

    ~Foo()
    {
        std::cout << "Deconstruct " << m_path_.get() << std::endl;
        // m_path_.reset();
    }

    Foo operator[](const std::string& key) &&
    {
        m_path_->push_back(key);
        return std::move(*this);
    }
    Foo operator[](const std::string& key) const&
    {
        Foo res(*this);
        res.m_path_->push_back(key);
        std::cout << res.str() << std::endl;

        return std::move(res);
    }
    std::string str() const
    {
        std::ostringstream os;
        if (m_path_ != nullptr)
        {
            for (auto&& item : *m_path_)
            {
                os << "/" << item;
            }
        }

        return os.str();
    }
    std::unique_ptr<std::list<std::string>> m_path_;
};
int main(int argc, char** argv)
{
    Foo v;
    std::cout << v["a"]["b"]["C"].str() << std::endl;
}