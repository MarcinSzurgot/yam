#pragma once

#include <iterator>

namespace yam {

template<
    typename Init,
    typename Cond,
    typename Updt
>
struct Algorit {
    using value_type = Init;
    using difference_type = std::ptrdiff_t;

    Algorit() : init_(), cond_(), updt_() { }

    Algorit(
        Init init,
        Cond cond,
        Updt updt
    ) : init_(init), cond_(cond), updt_(updt)
    {
    }

    Algorit& operator=(const Algorit& other) {
        init_ = other.init_;
        cond_ = other.cond_;
        updt_ = other.updt_;
        return *this;
    }

    auto operator*() const -> Init { return init_; }

    auto operator->() const -> const Init* { return std::addressof(init_); }
    auto operator->()       ->       Init* { return std::addressof(init_); }

    auto operator++() -> Algorit& {
        (*updt_)(init_);
        return *this;
    }

    auto operator++(int) -> Algorit {
        auto tmp = *this;
        ++tmp;
        return tmp;
    }

    friend auto operator==(
        const Algorit& lhs,
        const Algorit& rhs
    ) -> bool { return lhs.init_ == rhs.init_; }

    friend auto operator==(
        const Algorit& lhs,
        std::default_sentinel_t
    ) -> bool { return !(*lhs.cond_)(lhs.init_); }

    friend auto operator==(
        std::default_sentinel_t s,
        const Algorit& rhs
    ) -> bool { return rhs == s; }

private:
    Init init_;
    std::optional<Cond> cond_;
    std::optional<Updt> updt_;
};

}