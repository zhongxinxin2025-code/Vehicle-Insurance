---
name: fraud-decision-engine
description: "A downstream skill that takes fraud probability and claim features, then uses the OpenAI API to generate a natural language explanation of why the claim was flagged and a recommended action for the claims officer"
---

## When to use
This skill is triggered after the run-inference skill has completed and the fraud probability score and claim features are stored in agent state. It is used to provide a human-readable explanation of the model's prediction and actionable recommendations for claims officers.

## How to execute
1. Retrieve the fraud probability score and claim features from agent state.
2. Use the OpenAI API to generate a natural language explanation of the model's prediction.
3. Provide a recommended action for the claims officer based on the explanation.
4. Store the explanation and recommendation in agent state.

## Inputs from agent state
- fraud_probability: float — predicted fraud probability score for the new claim
- new_claim_data: pd.DataFrame — new insurance claim features

## Outputs to agent state
- explanation: str — natural language explanation of the model's prediction
- recommendation: str — recommended action for the claims officer

## Output format
The output should include the explanation and recommendation, all stored as named fields in AgentState.
