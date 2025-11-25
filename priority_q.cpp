#include <vector>
#include <iostream>
#include <utility>

class priority_queue {
    private:
        std::vector<std::pair<float, std::string>> priority;


    public:
        void push(float danger, const std::string& name) {
            int i = 0;
            while (i < priority.size() && priority[i].first >= danger) {
                i++;
            }

            priority.insert(priority.begin() + i, {danger, name});
        }

        std::pair<float, std::string> pop() {
            if (priority.empty()) {
                return {0,""};
            }

            std::pair<float, std::string> val = priority[0];
            priority.erase(priority.begin()); 
            return val;
        }

        bool isEmpty() {
            return priority.empty();
        }
};

int main() {
    priority_queue dangerous_items;
    dangerous_items.push(100, "truck");
    dangerous_items.push(70, "car1");
    dangerous_items.push(10, "cat");
    dangerous_items.push(70, "car2");
    dangerous_items.push(20, "dog");
    dangerous_items.push(70, "car3");

    while (!dangerous_items.isEmpty()) {
        std::cout << dangerous_items.pop().second << std::endl;
    }
}