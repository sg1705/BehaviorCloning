#Behavior Cloning

|  Learning Rate | Epochs | Steering Angle Correction (Radian) | % Flipped | Channels | Model | Driving Quality Track 1 | Time on Track2 | Validation Loss | Validation Accuracy | Test Score Accuracy |
|  :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
|  0.001 | 5 | 0.17 | 0.5 | 3 | Architecture A | Drunk | 1 minute | 0.014 | 0.175 | 0.186 |
|  0.0001 | 5 | 0.17 | 0.5 | 3 | Architecture A | Accident | 10 seconds | 0.0141 | 0.1773 | 0.186 |
|  0.01 | 5 | 0.17 | 0.5 | 3 | Architecture A | Accident | 4 seconds | 0.0416 | 0.1773 | 0.186 |
|  0.001 | 10 | 0.17 | 0.5 | 3 | Architecture A | Accident | 5 seconds | 0.0136 | 0.1742 | 0.186 |
|  0.0015 | 10 | 0.17 | 0.5 | 3 | Architecture A | Accident | 25 seconds | 0.0124 | 0.1803 | 0.186 |
|  0.0015 | 5 | 0.17 | 0.5 | 3 | Architecture A | Novice | 3 minutes | 0.01041 | 0.1856 | 0.186 |
|  0.0015 | 5 | 0.17 | 0.5 | 3 | Architecture A | Novice | 3 minutes | 0.0149 | 0.1773 | 0.186 |
|  0.002 | 5 | 0.17 | 0.5 | 3 | Architecture A | Novice++ | 5 seconds | 0.014 | 0.1818 | 0.186 |
|  0.002 | 5 | 0.2 | 0.5 | 3 | Architecture A | Accident | 5 seconds | 0.0144 | 0.178 | 0.186 |
|  0.002 | 5 | 0.12 | 0.5 | 3 | Architecture A | Accident | 5 seconds | 0.0158 | 0.1826 | 0.186 |
|  0.002 | 5 | 0.12 | 0.5 | 3 | Architecture B | Novice | 10 seconds | 0.145 | 0.1924 | 0.186 |
|  0.002 | 10 | 0.15 | 0.5 | 3 | Architecture B | Accident | 5 seconds | 0.0164 | 0.178 | 0.186 |
|  0.0015 | 10 | 0.15 | 0.5 | 3 | Architecture C | Accident | 5 seconds | 0.0377 | 0.178 | 0.186 |
|  0.0015 | 20 | 0.17 | 0.5 | 3 | Architecture A | Accident | 5 seconds | 0.153 | 0.1689 | 0.186 |
