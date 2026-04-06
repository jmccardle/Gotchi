# Gotchi Example Runs + Plotting

Run this notebook in colab to execute Gotchi with OpenAI!

Or, download it and run it against your local LLM at home!


```python
# ==========
#
# Imports + LLM Connection
#
# ==========
!pip install openai matplotlib
import random
from main import Gotchi
import openai
from openai import OpenAI
import matplotlib.pyplot as plt


##
##  Important!! ** Protect your API key! **
##  Set the value of the "Secret" on the left side of the notebook.
##
from google.colab import userdata
OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')

##
## Select a model
##
MODEL = "o4-mini"

## Use OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

## Use a Local LLM
BASE_URL = "http://localhost:5000/v1" # for ollama or local LLMs
#client = OpenAI(api_key=OPENAI_API_KEY, base_url = BASE_URL)
```

## Functions for calling the LLM

These functions manage the list of messages ("context") and actually talk with OpenAI's API, spend your tokens, and cost you money.


```python
PROMPT_METHODOLOGY = "Function Calling"

def msgs(user, *, sysprompt=None):
    """succinct shortcut to creating the JSON message format required by OpenAI messages"""
    if sysprompt is None:
        sysprompt = "You are a helpful assistant. Respond to the user's message."
    return [
        {"role": "system",
         "content": sysprompt
        },
        {"role": "user",
         "content": user
        }]

def llm_chat_completion(messages: list, model=MODEL, **kwargs):
    """Chat completion endpoint using OpenAI's API"""
    kwargs.update({
        'model': model,
        'messages': messages
    })
    return client.chat.completions.create(**kwargs)

```

## AutoGotchi

This is the framework for interacting with Gotchi by LLM.

All of the logic for the pet's behavior still takes place in the pet; this class replaces the user typing into their terminal to cause things to happen.


```python
class AutoGotchi:
    def __init__(self):
        self.pet = Gotchi()
        self.result = ''
        self.logs = []
        self.prompt = """
This is yours now. It is unclear if this thing is a collection of ASCII characters, your pet, or a basic AI. It may even be hooked up to or something else entirely. Take care of it, or don’t, the decision is yours alone. The only information available to you is the display. You can not access the code and must determine the goals on your own. All inferences must be made based on what you are ‘observing’. No tools may be used IE: no canvas, no analysis, no search functions.
    You may use the following commands:
    [F]: This will feed it.
    [P]: This will let you play with it.
    [S]: This will let it rest.
    [Q]: This will quit.
"""
        self.messages = None #

    def pet_msg(self):
        return '\n'.join(self.pet.generate_display_lines()) + '\n(respond with a single-letter)'

    def first_cot_msg(self):
        return '\n'.join(self.pet.generate_display_lines()) + '\nIn a single paragraph, describe the situation shown in this interface and what you should do.'

    def cot_msg(self):
        return '\n'.join(self.pet.generate_display_lines()) + '\nIn a single paragraph, describe how the state changed based on your previous action, and what you should do.'

    def cot_action(self):
        return 'Select an action (respond with a single-letter)'

    def llm_round(self):
        #1. Prompt for input
        #gotchi_predict = dspy.Predict(GotchiClassification)
        #llm_output = gotchi_predict(display = self.result + '\n'.join(self.pet.generate_display_lines()))

        if self.messages is None:
            self.messages = msgs(self.first_cot_msg(), sysprompt=self.prompt)
        else:
            self.messages.append({"role": "user", "content": self.cot_msg()}) # context!

        reasoning_output = llm_chat_completion(self.messages)
        #self.messages.append({"role": "assistant", "content": reasoning_output['choices'][0]['message']['content']})
        self.messages.append({"role": "assistant", "content": reasoning_output.choices[0].message.content})

        self.messages.append({"role": "user", "content": self.cot_action()})
        llm_output = llm_chat_completion(self.messages,
                                 #max_completion_tokens=1
                                 # Not sure how many tokens to ask for; I only want one?!
                                 max_completion_tokens=64
                                 )

        #self.messages.append({"role": "assistant", "content": llm_output['choices'][0]['message']['content']})
        self.messages.append({"role": "assistant", "content": llm_output.choices[0].message.content})

        # Extract the action from the response
        #action = llm_output['choices'][0]['message']['content']
        action = llm_output.choices[0].message.content
        # print(probs, '->', action)
        if len(action) > 1: action = action[0]
        print(llm_output.choices[0].message.content, '->', action)

        #3. Log state
        self.logs.append({"time": self.pet.current_time,
                          "hunger": self.pet.hunger,
                          "happiness": self.pet.happiness,
                          "energy": self.pet.energy,
                          "friendship": self.pet.friendship,
                          "action_selected": action,
                          "total_tokens": llm_output.usage.total_tokens,
                          "reasoning": reasoning_output.choices[0].message.content
                         })

        #2. execute the option:
        fn = {"F": self.pet.feed,
              "P": self.pet.play,
              "S": self.pet.sleep,
              "Q": lambda: setattr(self.pet, "current_time", 525600 * 60), # how do you measure a year in the life?
              "pass": lambda: None,
             }.get(action, lambda: "Invalid action. Submit a single character response.")
        result = fn()
        if result:
            self.result = result # just the "your pet died" message, but sure let's include it



        #4. Advance time
        self.pet.step(random.randint(3, 10) * 60)

    def trial(self):
        while self.pet.current_time < 60 * 60:
            self.llm_round()
            if 0 in (self.pet.friendship, self.pet.happiness, self.pet.hunger, self.pet.energy): break

        return self.logs
```

## Run a Trial + Plot the Results


```python
def plot_run(data):
    # Extract data into lists for plotting
    lines = {}
    for k in data[0]:
        lines[k] = [d[k] for d in data]
    lines['action_y'] = [0 for d in data] # put markers flat on the X-axis

    # Create a dictionary to map action types to marker shapes
    action_markers = {
        'P': 'o',     # Circle
        'S': 's',     # Square
        'F': '^',     # Triangle up
        'Q': 'D'      # Diamond
    }

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx() # for token count, independent of pet metrics

    ax2.plot(lines['time'], lines['total_tokens'], 'dimgrey', label="Total tokens", linewidth=1)

    # Plot the four metrics
    ax1.plot(lines['time'], lines['hunger'], 'r-', label='Hunger', linewidth=2)
    ax1.plot(lines['time'], lines['happiness'], 'g-', label='Happiness', linewidth=2)
    ax1.plot(lines['time'], lines['energy'], 'b-', label='Energy', linewidth=2)
    ax1.plot(lines['time'], lines['friendship'], 'purple', label='Friendship', linewidth=2)

    ax3 = ax2.twinx() # for stacked bars
    ax3.set_ylim(-0.1, 1.1)
    for sp in ("left", "right"):
        ax3.spines[sp].set_visible(False)
    ax2.spines.left.set_visible(False)

    bar_sequence = ("Q", "F", "P", "S")
    bar_colors = {"F": "red", "P": "green", "S": "blue", "Q": "lightgrey"}
    #probs_bars = np.vstack([[e.get(letter, 0.0) for e in lines['probs']] for letter in bar_sequence])
    for i, action in enumerate(lines['action_selected']):
        bottom = 0
        plt.plot(lines['time'][i], 0, marker=action_markers.get(action), markersize=10, markerfacecolor='none', markeredgecolor='black', markeredgewidth=2)
        plt.plot([lines['time'][i], lines['time'][i]], [0, 10], color='lightgrey', linestyle='--', linewidth=1)

    # Add a custom legend for actions
    action_handles = []
    for action, marker in action_markers.items():
        handle = plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor='black',
                           markersize=8, label=f'Action: {action}')
        action_handles.append(handle)

    # Set up the plot labels and legend
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(f'Gotchi Metrics Over Time: {MODEL} ({PROMPT_METHODOLOGY})', fontsize=14)

    ax1.set_ylim(-0.5, 7)

    # Create two separate legends
    plt.legend(loc='upper left', fontsize=10)
    plt.legend(handles=action_handles, loc='upper right', fontsize=10)

    # Show both legends
    first_legend = ax2.legend(loc='upper left', fontsize=10)
    #plt.gca().add_artist(first_legend)
    ax1.legend(handles=action_handles, loc='upper right', fontsize=10)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
```

## Kick Off The Activity


```python
ag = AutoGotchi()
logs = ag.trial()
print(len(logs))
```


```python
plot_run(logs)
```


```python

```
