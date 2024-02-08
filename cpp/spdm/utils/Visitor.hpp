/**
 * @file visitor.h
 *
 *  created on: 2014-4-23
 *      Author: salmon
 */

#ifndef VISITOR_H_
#define VISITOR_H_

#include <memory>
#include <string>

namespace simpla {
/**
 * @ingroup  design_pattern
 *
 * @addtogroup vistor Vistor
 * @{
 **/
/**
 *  \brief  Generic Visitor
 *
 *  Double Visitor pattern :
 *  purpose: pass variadic parameters to acceptor
 *  visitor visit acceptor twice, first get acceptor type , second get parameters type .
 *  \code
 *   struct Foo1: public Acceptor
 *   {
 *   	typedef Foo1 this_type;
 *
 *   	virtual bool CheckType(std::type_info const &t_info)
 *   	{
 *   		return typeid(this_type) == t_info;
 *   	}
 *
 *   	template<typename ...Args>
 *   	void filter(Visitor<this_type, Args...> &visitor)
 *   	{
 *   		visitor.excute([this](Args ... args)
 *   		{	this->Command(std::forward<Args>(args)...);});
 *   	}
 *   	void filter(Visitor<this_type, const char *> &visitor)
 *   	{
 *   		if (visitor.GetName() == "Command2")
 *   		{
 *   			visitor.excute([this](std::string const & args)
 *   			{	this->Command2(args);});
 *   		}
 *   		else
 *   		{
 *   			std::cout << "Unknown function name!" << std::endl;
 *   		}
 *   	}
 *
 *   	void Command2(std::string const & s)
 *   	{
 *   		std::cout << "This is Foo1::Command2(string). args=" << s << std::endl;
 *   	}
 *
 *   	void Command(int a,  unsigned int  b)
 *   	{
 *   		std::cout << "This is Foo1::Command(unsigned int ,int). args=" << a << "     " << b << std::endl;
 *   	}
 *
 *   	template<typename ... Args>
 *   	void Command(Args const & ...args)
 *   	{
 *   		std::cout << "This is Foo1::Command(args...). args=";
 *
 *   		print(args...);
 *
 *   		std::cout << std::endl;
 *   	}
 *
 *   	void print()
 *   	{
 *   	}
 *
 *   	template<typename T, typename ... Others>
 *   	void print(T const &v, Others && ... others)
 *   	{
 *   		std::cout << v << " ";
 *   		print(std::forward<Others >(others )...);
 *   	}
 *
 *   };
 *
 *    unsigned int  main(int argc, char **argv)
 *   {
 *   	Acceptor * f1 = dynamic_cast<Acceptor*>(new Foo1);
 *   	auto v1 = createVisitor<Foo1>("Command1", 5, 6);
 *   	auto v2 = createVisitor<Foo1>("Command2", "hello world");
 *   	auto v3 = createVisitor<Foo1>("Command3", 5, 6, 3);
 *   	f1->filter(v1);
 *   	f1->filter(v2);
 *   	f1->filter(v3);
 *
 *   	delete f1;
 *
 *   }
 *  \endcode
 *
 */
struct Acceptor;
template <typename T>
class Visitor;
struct VisitorInterface {
   protected:
    VisitorInterface() {}

   public:
    virtual ~VisitorInterface() {}

    template <typename TObj>
    void visit(Acceptor &p) {
        auto *self = dynamic_cast<Visitor<TObj> *>(this);
        if (self != nullptr) {}
    }
    void visit(Acceptor const &p) {}
    void visit(Acceptor &p) const {}
    void visit(Acceptor const &p) const {}
};
template <typename T>
class Visitor : public VisitorInterface {
   protected:
    Visitor() {}

   public:
    virtual ~Visitor() {}
    std::shared_ptr<VisitorInterface> New();
};

template <typename TObj>
struct Acceptor {
    Acceptor() {}

    virtual ~Acceptor() {}

    virtual void accept(Visitor &visitor) { visitor.visit(*this); }

    virtual void accept(Visitor const &visitor) { visitor.visit(*this); }

    virtual void accept(Visitor &visitor) const { visitor.visit(*this); }

    virtual void accept(Visitor const &visitor) const { visitor.visit(*this); }
};

//    template<typename TFUN>
//    inline void execute(TFUN const &f)
//    {
//        callFunc(f, typename make_int_sequence<sizeof...(Args)>::value_type_info());
//    }
//
// private:
//// Unpack tuple to args...
////\note  http://stackoverflow.com/questions/7858817/unpacking-a-tuple-to-call-a-matching-function-pointer
//
//
//    template<typename TFUN, unsigned int  ...S>
//    inline void callFunc(TFUN const &fun, integer_sequence<int, S...>)
//    {
//        fun(std::Serialize<S>(args_) ...);
//    }

/** @}*/
}  // namespace simpla

#endif /* VISITOR_H_ */
