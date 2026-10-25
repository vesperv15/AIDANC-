#ifndef AIDAN_CONFIG_HPP
#define AIDAN_CONFIG_HPP

#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <cstdint>
#include <memory>

namespace aidan::config {
    //model ve vektör boyutları
constexpr int32_t EMBED_DIM = 32;
constexpr int32_t HIDDEN_DIM = 128;
constexpr int32_t FOCUS_DIM =32;
constexpr int32_t PREDICTION_DIM = 50;
constexpr int32_t MAX_SEQ_LEN = 128;

//reinforcement learning opti. hiperparametreleri
constexpr float GAMMA = 0.99f;
constexpr float LR= 0.0003f;
constexpr float ENTROPY_BETA = 0.08f;
constexpr float VALUE_LOSS_BETA = 0.5f;
constexpr float INTRINSIC_BETA = 0.6f;
constexpr float PREDICTION_LOSS_BETA = 0.1f;
constexpr int32_t MAX_STEPS = 30;
constexpr int32_t DEFAULT_NUM_EPISODES = 7000;
constexpr float EPSILON_START = 1.0f;
constexpr float EPSILON_END = 0.05f;
constexpr float EPSILON_DECAY = 5000.0f;
constexpr int32_t TARGET_UPDATE_FREQ = 100;
constexpr size_t MAX_BANK_SIZE = 50000; // 15gb ram için sınrı

//eylem tipleri
enum class ActionType : uint8_t {
    SIMPLIFY = 0,
    EXPAND,
    FACTOR,
    DIFFERENTIATE_X,
    TERMINATE,
    UNDO,
    COUNT
};

inline const std::vector<std::string> ACTION_TOKENS = {
    "SIMPLIFY", "EXPAND", "FACTOR", "DIFFERENTIATE_X", "TERMINATE", "UNDO"
};

inline const std::vector<std::string> MATH_TOKENS = {
    "x", "y", "z", "t",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "+", "-", "*", "/", "**",
    "(", ")",
    "sin", "cos", "exp", "log", "tan",
    "=", "WRT", "^"
};
//singleton sözlük yapısı
class Vocabulary {
public:
    static const Vocabulary& instance() {
        static Vocabulary inst;
        return inst;
    }

    int32_t get_id(const std::string& token) const {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) return it->second;
        return unk_id_;
    }

    std::string get_token(int32_t id) const {
        if (id >= 0 && id < static_cast<int32_t>(id_to_token_.size())) {
            return id_to_token_[id];
        }
        return "<UNK>";
    }

    size_t size() const noexcept { return id_to_token_.size(); }
    int32_t pad_id() const noexcept { return pad_id_; }
    int32_t unk_id() const noexcept { return unk_id_; }

private:
    Vocabulary() {
        for (const auto& tok : MATH_TOKENS) add_token(tok);
        for (const auto& act : ACTION_TOKENS) add_token(act);
        unk_id_ = add_token("<UNK>");
        pad_id_ = add_token("<PAD>");
    }

    int32_t add_token(const std::string& tok) {
        int32_t id = static_cast<int32_t>(id_to_token_.size());
        id_to_token_.push_back(tok);
        token_to_id_[tok] = id;
        return id;
    }

    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    int32_t unk_id_{0};
    int32_t pad_id_{0};
};
} //namespace aidan
#endif