#include "EntryXML.h"
#include "Entry.h"
#include "Node.h"
#include <any>
#include <map>
#include <pugixml.hpp>
#include <vector>
namespace sp
{

    EntryTmpl<entry_tag_xml>::EntryTmpl() {}

    EntryTmpl<entry_tag_xml>::EntryTmpl(EntryTmpl const &other) {}

    EntryTmpl<entry_tag_xml>::EntryTmpl(EntryTmpl &&other) {}

    EntryTmpl<entry_tag_xml>::~EntryTmpl() {}

    EntryTmpl<entry_tag_xml> &EntryTmpl<entry_tag_xml>::operator=(EntryTmpl const &other) {}

    void EntryTmpl<entry_tag_xml>::swap(EntryTmpl &other){}

    Entry *EntryTmpl<entry_tag_xml>::copy() const{}

    Entry *EntryTmpl<entry_tag_xml>::move(){}

    std::ostream &EntryTmpl<entry_tag_xml>::repr(std::ostream &os) const {} // represent object as string and push ostream

    int EntryTmpl<entry_tag_xml>::type() const {}

    //----------------------------------------------------------------------------------------------------------
    // attribute
    //----------------------------------------------------------------------------------------------------------
    bool EntryTmpl<entry_tag_xml>::has_attribute(std::string const &k) const {} // if key exists then return true else return false

    bool EntryTmpl<entry_tag_xml>::check_attribute(std::string const &k, std::any const &v) const {} // if key exists and value ==v then return true else return false

    std::any EntryTmpl<entry_tag_xml>::get_attribute(std::string const &key) const {} // get attribute at key, if key does not exist return nullptr

    std::any EntryTmpl<entry_tag_xml>::get_attribute(std::string const &key, std::any const &default_value = std::any{}){} // get attribute at key, if key does not exist return nullptr

    void EntryTmpl<entry_tag_xml>::set_attribute(std::string const &key, std::any const &v){} // set attribute at key as v

    void EntryTmpl<entry_tag_xml>::remove_attribute(std::string const &key = ""){} // remove attribute at key, if key=="" then remove all

    Range<Iterator<const std::pair<const std::string, std::any>>> EntryTmpl<entry_tag_xml>::attributes() const {} // return reference of  all attributes

    //----------------------------------------------------------------------------------------------------------
    // as leaf node,  need node.type = Scalar || Block
    //----------------------------------------------------------------------------------------------------------

    std::any EntryTmpl<entry_tag_xml>::get_scalar() const {} // get value , if value is invalid then throw exception

    void EntryTmpl<entry_tag_xml>::set_scalar(std::any const &){} // set value , if fail then throw exception

    std::tuple<std::shared_ptr<char>, std::type_info const &, std::vector<size_t>>
    EntryTmpl<entry_tag_xml>::get_raw_block() const {} // get block

    void EntryTmpl<entry_tag_xml>::set_raw_block(std::shared_ptr<char> const &, std::type_info const &, std::vector<size_t> const &){} // set block

    //----------------------------------------------------------------------------------------------------------
    // as Hierarchy tree node
    // function level 0
    std::shared_ptr<const Node> EntryTmpl<entry_tag_xml>::find_child(std::string const &) const {} // return reference of child node , if key does not exists then insert new

    std::shared_ptr<Node> EntryTmpl<entry_tag_xml>::find_child(std::string const &){} // return reference of child node , if key does not exists then insert new

    std::shared_ptr<Node> EntryTmpl<entry_tag_xml>::append(){} // append node to tail of list , return reference of new node

    std::shared_ptr<Node> EntryTmpl<entry_tag_xml>::append(std::shared_ptr<Node> const &){}

    void EntryTmpl<entry_tag_xml>::append(const Iterator<std::shared_ptr<Node>> &b, const Iterator<std::shared_ptr<Node>> &){} // insert node to object

    std::shared_ptr<Node> EntryTmpl<entry_tag_xml>::insert(std::string const &key, std::shared_ptr<Node> const &n = nullptr){} // insert node to object

    void EntryTmpl<entry_tag_xml>::insert(Iterator<std::pair<const std::string, std::shared_ptr<Node>>> const &b,
                                          Iterator<std::pair<const std::string, std::shared_ptr<Node>>> const &e){} // insert node to object

    std::shared_ptr<Node> EntryTmpl<entry_tag_xml>::child(std::string const &key){} // get child, create new if key does not  exist

    // std::shared_ptr<const Node> child(std::string const &key) const {} // get child, create new if key does not  exist

    std::shared_ptr<Node> EntryTmpl<entry_tag_xml>::child(int idx){} // return reference i-th child node , if idx does not exists then throw exception

    // std::shared_ptr<const Node> child(int idx) const {} // return reference i-th child node , if idx does not exists then throw exception

    void EntryTmpl<entry_tag_xml>::remove_child(std::string const &key){} // remove child at key

    void EntryTmpl<entry_tag_xml>::remove_child(int idx){} // remove i-th child

    void EntryTmpl<entry_tag_xml>::remove_children(){} // remove children , set node.type => Null

    // std::pair<Iterator<const Node>, Iterator<const Node>> children() const {} // reutrn list of children

    std::pair<Iterator<Node>, Iterator<Node>>
    EntryTmpl<entry_tag_xml>::children(){} // reutrn list of children

    // level 1
    // std::pair<Iterator<const Node>, Iterator<const Node>> select(XPath const &path) const {} // select from children

    std::pair<Iterator<Node>, Iterator<Node>>
    EntryTmpl<entry_tag_xml>::select(XPath const &path){} // select from children

    // std::shared_ptr<const Node> select_one(XPath const &path) const {} // return the first selected child

    std::shared_ptr<Node>
    EntryTmpl<entry_tag_xml>::select_one(XPath const &path){} // return the first selected child

} // namespace sp
