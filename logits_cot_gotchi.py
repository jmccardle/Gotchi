# ==========
#
# Imports + LLM Connection
#
# ==========

import random
from gotchi import Gotchi
import openai
from openai import OpenAI

client = OpenAI(api_key='your_openai_api_key', base_url = 'http://192.168.1.100:5000/v1')

# Set your OpenAI API key and base URL
# TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url='https://api.openai.com/v1')'
# openai.api_base = 'https://api.openai.com/v1'  # Default OpenAI API base URL



MODEL = "qwen2.5-coder-32b"
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

#

# Remove logits related functions and variables

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
                                 max_tokens=1
                                 )

        #self.messages.append({"role": "assistant", "content": llm_output['choices'][0]['message']['content']})
        self.messages.append({"role": "assistant", "content": llm_output.choices[0].message.content})

        # Extract the action from the response
        #action = llm_output['choices'][0]['message']['content']
        action = llm_output.choices[0].message.content
        # print(probs, '->', action)

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
             }[action]
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

import matplotlib.pyplot as plt
#import numpy as np

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
        # bar plot for probs at this decision
        bottom = 0
        #for char in bar_sequence:
        #    this_height = lines['probs'][i].get(char, 0.0) # the height of the bar is this action's probability
        #    ax3.bar(lines['time'][i], this_height, bottom=bottom, width=50, alpha=0.3, color=bar_colors[char], edgecolor='none') #label=f'Probability of {char!r}'
        #    bottom += this_height # The next bar in this stack will be above the previous one
        # marker at position (time, 0)
        plt.plot(lines['time'][i], 0, marker=action_markers.get(action), markersize=10, markerfacecolor='none', markeredgecolor='black', markeredgewidth=2)
        plt.plot([lines['time'][i], lines['time'][i]], [0, 10], color='lightgrey', linestyle='--', linewidth=1)
        ## TODO: Plot a vertical line from (lines['time'][i], 0) to (lines['time'][i], 10)
    #ax1.plot(lines['time'], lines['action_y'], marker=[action_markers.get(c) for c in lines['action_selected']], markersize=10, markerfacecolor='none', markeredgecolor='black', markeredgewidth=2)



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

    #plt.ylim(-0.5, 7)
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

if __name__ == '__main__':
    ag = AutoGotchi()
    logs = ag.trial()
    #logs = [{'time': 0, 'hunger': 5.0, 'happiness': 5.0, 'energy': 5.0, 'friendship': 5.0, 'probs': {'F': 0.28243510724023047, 'P': 0.6747860244928637, 'S': 0.03337279209551519, '[P': 0.009406076171390573}, 'action_selected': 'P', 'total_tokens': 273}, {'time': 300, 'hunger': 3.8000000000000003, 'happiness': 5.0, 'energy': 3.75, 'friendship': 5.000000000000001, 'probs': {'F': 0.17561439783233646, 'H': 0.0010042371213737715, 'P': 0.7279014730741441, 'S': 0.09547989197214564}, 'action_selected': 'P', 'total_tokens': 375}, {'time': 900, 'hunger': 2.0, 'happiness': 4.5, 'energy': 3.5, 'friendship': 4.900000000000002, 'probs': {'F': 0.540020481287321, 'H': 0.0008986779927799087, 'P': 0.4205683737011625, 'S': 0.03851246701873647}, 'action_selected': 'F', 'total_tokens': 477}, {'time': 1200, 'hunger': 3.9, 'happiness': 4.0, 'energy': 2.75, 'friendship': 5.000000000000003, 'probs': {'F': 0.36967262891417574, 'H': 0.001233052236737488, 'P': 0.5907351789761418, 'S': 0.0383591398729449}, 'action_selected': 'P', 'total_tokens': 594}, {'time': 1500, 'hunger': 2.6999999999999997, 'happiness': 4.0, 'energy': 1.5, 'friendship': 5.0000000000000036, 'probs': {'F': 0.413458591567274, 'P': 0.18929513693060185, 'Q': 0.002721327326399317, 'S': 0.3945249441757249}, 'action_selected': 'F', 'total_tokens': 696}, {'time': 1740, 'hunger': 3.6999999999999997, 'happiness': 4.0, 'energy': 1.25, 'friendship': 5.200000000000004, 'probs': {'F': 0.6369383242091232, 'P': 0.0992158548470377, 'Q': 0.07848620121267876, 'S': 0.18535961973116036}, 'action_selected': 'F', 'total_tokens': 801}, {'time': 2100, 'hunger': 2.8999999999999995, 'happiness': 2.5, 'energy': 1.0, 'friendship': 5.100000000000005, 'probs': {'F': 0.5931561076170911, 'P': 0.2673564040656876, 'Q': 0.0050522457802350445, 'S': 0.1344352425369862}, 'action_selected': 'F', 'total_tokens': 903}, {'time': 2280, 'hunger': 5.8999999999999995, 'happiness': 2.5, 'energy': 0.75, 'friendship': 5.300000000000005, 'probs': {'F': 0.6351664007744389, 'P': 0.030650134261018376, 'Q': 0.02462807476532874, 'S': 0.309555390199214}, 'action_selected': 'F', 'total_tokens': 1008}, {'time': 2580, 'hunger': 5.7, 'happiness': 1.5, 'energy': 1.0, 'friendship': 5.300000000000006, 'probs': {'F': 0.6394945260491854, 'P': 0.07402687112885839, 'Q': 0.007104143111265167, 'S': 0.27937445971069097}, 'action_selected': 'F', 'total_tokens': 1123}, {'time': 3000, 'hunger': 5.500000000000001, 'happiness': 0.5, 'energy': 1.25, 'friendship': 5.300000000000007, 'probs': {'F': 0.9001872587882966, 'P': 0.016487508080926323, 'Q': 0.003429074871958939, 'S': 0.07989615825881805}, 'action_selected': 'F', 'total_tokens': 1241}]
    plot_run(logs)
