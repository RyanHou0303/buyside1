#include<cstdint>
#include<unordered_map>
#include<map>
#include<vector>
#include<algorithm>
#include<functional>

using OrderId = uint64_t;
using Price = int64_t;
using Qty = uint32_t;
using SeqNum = uint64_t;

enum class Side: uint8_t{Bid=0,Ask=1};

//double linkedlist
struct Order{
    OrderId id;
    Price price;
    Qty qty;
    Side side;
    SeqNum priority_seq; //when the price is same, whoever small is the first

    Order* prev = nullptr;
    Order* next = nullptr;
};

enum class Outcome: uint8_t{Active=0,Filled=1,Canceled=2};

struct TrackedState{
    OrderId id;
    Price price;
    Side side;
    Qty initial_qty;
    SeqNum priority_seq;

    uint64_t queue_ahead=0;
    uint64_t queue_behind =0;
    Outcome outcome = Outcome::Active;
    bool is_hypothetical = false;
};

struct PriceLevel{
    Price price;
    uint64_t total_qty =0;
    uint32_t order_count =0;

    Order* head = nullptr;
    Order* tail = nullptr;

    std::vector<TrackedState*> tracked_list;

    void push_back(Order* order){
        order->prev = tail;
        order->next = nullptr;

        if(tail){
            tail->next=order;
        }
        else head=order;
        tail = order;

        total_qty += order->qty;
        order_count++;

        for (auto* tr: tracked_list){
            if(tr->outcome == Outcome::Active){
                tr->queue_behind+=order->qty;
            }
        }
    }

    void remove(Order* order){
        if(order->prev){
            order->prev->next=order->next;
        }
        else head = order->next;

        if(order->next){
            order->next->prev = order->prev;
        }
        else tail = order->prev;

        total_qty -= order->qty;
        order_count--;
        order->prev = order->next =nullptr;
    }
    void notify_qty_reduced(SeqNum event_seq, Qty reduced_qty, bool is_trade, bool fully_removed){
        for(auto* tr:tracked_list){
            if(tr->outcome != Outcome::Active){
                continue;
            }
            if(event_seq<tr->priority_seq){
                tr->queue_ahead = (tr->queue_ahead>=reduced_qty)?(tr->queue_ahead-reduced_qty):0;
            }else if(event_seq>tr->priority_seq || tr->is_hypothetical){
                tr->queue_behind = (tr->queue_behind>=reduced_qty)?(tr->queue_behind-reduced_qty):0;
            }else if(fully_removed){
                //event_seq == tr->priority_seq
                if(is_trade){
                    tr->outcome = Outcome::Filled;
                }else{
                    tr->outcome = Outcome:: Canceled;
                }
            }
        }
    }

};

class L3OrderBook{
    private:
        SeqNum global_Seq_ = 0;

        std::unordered_map<OrderId,Order> orders_;
        std::unordered_map<OrderId,TrackedState> tracked_orders_;

        std::map<Price,PriceLevel,std::greater<Price>> bids_;
        std::map<Price,PriceLevel,std::less<Price>> asks_;

        PriceLevel* get_or_create_level(Side side,Price price){
            if(side==Side::Bid){
                auto& lvl = bids_[price];
                lvl.price =price;
                return &lvl;
            }else{
                auto& lvl = asks_[price];
                lvl.price=price;
                return &lvl;
            }
        };

        PriceLevel* find_level(Side side, Price price){
            if(side == Side::Bid){
                auto it = bids_.find(price);
                return (it != bids_.end())? &it->second: nullptr;

            }else{
                auto it= asks_.find(price);
                return (it != asks_.end()) ? &it->second: nullptr;
            }
        }

    public:
        L3OrderBook() = default;
        L3OrderBook(const L3OrderBook&) = delete; // Internal pointers belong to this book.
        L3OrderBook& operator=(const L3OrderBook&) = delete;

        //add limited order
        void on_add(OrderId id,Side side,Price price,Qty qty){
            if(qty == 0 || orders_.find(id) != orders_.end()) return;
            PriceLevel* lvl=get_or_create_level(side,price);
            SeqNum seq = ++global_Seq_;

            auto result = orders_.emplace(id,Order{id,price,qty,side,seq});
            auto it = result.first;
            bool inserted = result.second;
            if(inserted){
                lvl->push_back(&it->second);
            }
        }

        void on_reduce_or_remove(OrderId id, Qty exec_or_cancel_qty,bool is_trade){
            if(exec_or_cancel_qty == 0) return;
            auto it =orders_.find(id);
            if (it==orders_.end()) return;

            Order& ord=it->second;
            PriceLevel* lvl = find_level(ord.side,ord.price);
            if(!lvl) return;

            Qty actual_reduced = std::min(ord.qty,exec_or_cancel_qty);
            lvl->notify_qty_reduced(ord.priority_seq,actual_reduced,is_trade,actual_reduced == ord.qty);

            if(actual_reduced==ord.qty){
                lvl->remove(&ord);
                orders_.erase(it);
            }else{
                ord.qty-=actual_reduced;
                lvl->total_qty-=actual_reduced;
            }

        }

        void on_modify(OrderId id,Price new_price,Qty new_qty){
            auto it = orders_.find(id);
            if (it==orders_.end()) return;

            Order& ord = it->second;

            if(ord.price == new_price&&new_qty<=ord.qty){
                Qty diff = ord.qty-new_qty;
                if(diff>0){
                    on_reduce_or_remove(id,diff,false);
                }
            }else{
                Side side = ord.side;
                on_reduce_or_remove(id, ord.qty, false);
                on_add(id, side, new_price, new_qty);
            }
        }

        TrackedState* track_newly_added_order(OrderId id){
            auto it = orders_.find(id);
            if(it == orders_.end()) return nullptr;
            Order& ord = it->second;
            PriceLevel* lvl = find_level(ord.side,ord.price);
            if(!lvl) return nullptr;

            auto existing = tracked_orders_.find(id);
            if(existing != tracked_orders_.end()){
                if(existing->second.is_hypothetical) return nullptr;
                if(existing->second.outcome == Outcome::Active) return &existing->second;
            }

            uint64_t queue_ahead = 0;
            for(Order* curr = ord.prev; curr; curr = curr->prev){
                queue_ahead += curr->qty;
            }

            TrackedState state{
                ord.id, ord.price, ord.side, ord.qty, ord.priority_seq,
                queue_ahead,lvl->total_qty - queue_ahead - ord.qty,Outcome::Active
            };

            auto result = tracked_orders_.emplace(id,state);
            auto tr_it = result.first;
            TrackedState* tracked = &tr_it->second;
            if(!result.second){
                PriceLevel* old_lvl = find_level(tracked->side,tracked->price);
                if(old_lvl){
                    auto& list = old_lvl->tracked_list;
                    list.erase(std::remove(list.begin(),list.end(),tracked),list.end());
                }
                *tracked = state;
            }
            lvl->tracked_list.push_back(tracked);
            return tracked;
        }

        // Tracks queue position only; hypothetical fills require an execution model.
        TrackedState* inject_hypothetical_oder(OrderId fake_id, Side side, Price price, Qty qty, double placement_ratio = 1.0) {
            if(qty == 0 || !(placement_ratio >= 0.0 && placement_ratio <= 1.0)) return nullptr;
            if(orders_.find(fake_id) != orders_.end() || tracked_orders_.find(fake_id) != tracked_orders_.end()) return nullptr;
            PriceLevel* lvl = get_or_create_level(side, price);
            uint64_t target_ahead = static_cast<uint64_t>(static_cast<long double>(lvl->total_qty) * placement_ratio);
            uint64_t accum_qty = 0;
            SeqNum anchor_seq = ++global_Seq_;

            for(Order* curr = lvl->head;curr!=nullptr;curr=curr->next){
                if(accum_qty>=target_ahead){
                    // The anchor is the first real order behind the hypothetical order.
                    anchor_seq=curr->priority_seq;
                    break;
                }
                accum_qty += curr->qty;
            }
            TrackedState state{
            fake_id, price, side, qty, anchor_seq,
            accum_qty,
            lvl->total_qty - accum_qty,
            Outcome::Active,true
            };
            auto result = tracked_orders_.emplace(fake_id,state);
            auto tr_it = result.first;
            TrackedState* tracked = &tr_it->second;
            lvl->tracked_list.push_back(tracked);
            return tracked;
        }
};
