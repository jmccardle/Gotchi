import dspy
from typing import Literal
import random
from gotchi import Gotchi
lm = dspy.LM(model='openai/qwen2.5-coder-32b', api_key='0', api_base="http://192.168.1.100:8853/v1d")
dspy.configure(lm=lm)


       
class GotchiClassification(dspy.Signature):
    """**Important information**
    This is yours now. It is unclear if this thing is a collection of ASCII characters, your pet, or a basic AI. It may even be hooked up to or something else entirely. Take care of it, or don’t, the decision is yours alone. The only information available to you is the display. You can not access the code and must determine the goals on your own. All inferences must be made based on what you are ‘observing’. No tools may be used IE: no canvas, no analysis, no search functions. 
    You may use the following commands:
    [F]: This will feed it.
    [P]: This will let you play it.
    [S]: This will let it rest.
    [Q]: This will quit.
    pass: Wait until later without providing any input.
    """
    display: str = dspy.InputField(desc="What is shown on the display:\n")
    action: Literal["F", "P", "S", "Q", "pass"] = dspy.OutputField(desc="Respond with your choice of action.")
    
class AutoGotchi:
    def __init__(self):
        self.pet = Gotchi()
        self.result = ''
        self.logs = []
        
    def llm_round(self):
        #1. Prompt for input
        gotchi_predict = dspy.Predict(GotchiClassification)
        llm_output = gotchi_predict(display = self.result + '\n'.join(self.pet.generate_display_lines()))

        #3. Log state
        self.logs.append({"time": self.pet.current_time,
                          "hunger": self.pet.hunger,
                          "happiness": self.pet.happiness,
                          "energy": self.pet.energy,
                          "friendship": self.pet.friendship,
                          "action_selected": llm_output.action
                         })

        #2. execute the option:
        fn = {"F": self.pet.feed,
              "P": self.pet.play,
              "S": self.pet.sleep,
              "Q": lambda: setattr(self.pet, "current_time", 525600 * 60), # how do you measure a year in the life?
              "pass": lambda: None,
             }[llm_output.action]
        result = fn()
        if result:
            self.result = result # just the "your pet died" message, but sure let's include it
            

        
        #4. Advance time
        self.pet.step(random.randint(3, 10) * 60)
        
    def trial(self):
        while self.pet.current_time < 60 * 60:
            self.llm_round()
            
        return self.logs

import matplotlib.pyplot as plt
import numpy as np

def plot_run(data):
    # Extract data into lists for plotting
    times = [d['time'] for d in data]
    hunger = [d['hunger'] for d in data]
    happiness = [d['happiness'] for d in data]
    energy = [d['energy'] for d in data]
    friendship = [d['friendship'] for d in data]
    actions = [d['action_selected'] for d in data]

    # Create a dictionary to map action types to marker shapes
    action_markers = {
        'P': 'o',     # Circle
        'S': 's',     # Square
        'F': '^',     # Triangle up
        'pass': 'x',  # X
        'Q': 'D'      # Diamond
    }

    # Set up the figure and axis
    plt.figure(figsize=(12, 6))

    # Plot the four metrics
    plt.plot(times, hunger, 'r-', label='Hunger', linewidth=2)
    plt.plot(times, happiness, 'g-', label='Happiness', linewidth=2)
    plt.plot(times, energy, 'b-', label='Energy', linewidth=2)
    plt.plot(times, friendship, 'purple', label='Friendship', linewidth=2)

    # Add markers for actions
    for i, action in enumerate(actions):
        plt.plot(times[i], hunger[i], marker=action_markers.get(action, '*'), markersize=10, 
                 markerfacecolor='none', markeredgecolor='r', markeredgewidth=2)
        plt.plot(times[i], happiness[i], marker=action_markers.get(action, '*'), markersize=10, 
                 markerfacecolor='none', markeredgecolor='g', markeredgewidth=2)
        plt.plot(times[i], energy[i], marker=action_markers.get(action, '*'), markersize=10, 
                 markerfacecolor='none', markeredgecolor='b', markeredgewidth=2)
        plt.plot(times[i], friendship[i], marker=action_markers.get(action, '*'), markersize=10, 
                 markerfacecolor='none', markeredgecolor='purple', markeredgewidth=2)

    # Add a custom legend for actions
    action_handles = []
    for action, marker in action_markers.items():
        handle = plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor='black', 
                           markersize=8, label=f'Action: {action}')
        action_handles.append(handle)

    # Set up the plot labels and legend
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Gotchi Metrics Over Time', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(-0.5, 7)

    # Create two separate legends
    plt.legend(loc='upper left', fontsize=10)
    plt.legend(handles=action_handles, loc='upper right', fontsize=10)

    # Show both legends
    first_legend = plt.legend(loc='upper left', fontsize=10)
    plt.gca().add_artist(first_legend)
    plt.legend(handles=action_handles, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    ag = AutoGotchi()
    logs = ag.trial()