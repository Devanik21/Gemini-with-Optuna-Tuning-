import streamlit as st
import google.generativeai as genai
import optuna

# --- Sidebar Configuration ---
st.sidebar.title("⚙️ Configuration")
api_key = st.sidebar.text_input("Google AI API Key", type="password", help="Get your key from https://aistudio.google.com/app/apikey")

# --- Initialize Gemini ---
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        st.session_state.model = model
        st.sidebar.success("✨ Gemini model configured!")
    except Exception as e:
        st.sidebar.error(f"Invalid API Key: {e}")

# --- Sample Prompt Testing ---
st.title("🔮 Gemini Prompt Optimizer for Next Word Prediction")

text_input = st.text_input("Enter partial sentence:", "The stars are")

# --- Prompt Candidates ---
prompt_templates = [
    "Predict the next word: '{}'",
    "What comes next after '{}'",
    "Continue the phrase: '{}'",
    "'{}' then?",
    "Next word after '{}':",
]

# --- Define Objective Function for Optuna ---
def objective(trial):
    template = trial.suggest_categorical("template", prompt_templates)
    full_prompt = template.format(text_input)

    try:
        response = model.generate_content(full_prompt)
        output = response.text.strip().split()
        if len(output) > 0:
            score = len(output[0])  # Simulated quality: shorter = better
        else:
            score = 10
    except Exception as e:
        score = 10  # Penalize error
    return score

# --- Run Optuna Optimization ---
if st.button("🔍 Tune Best Prompt"):
    with st.spinner("Tuning prompt with Optuna..."):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=10)
        best_template = study.best_params["template"]
        final_prompt = best_template.format(text_input)
        final_output = model.generate_content(final_prompt).text.strip()
        st.success(f"✨ Best Prompt: {best_template}")
        st.write("📝 Gemini's next word prediction:")
        st.markdown(f"**{final_output}**")

# Footer
st.markdown("---")
st.caption("💖 Built with Gemini, Optuna & a sprinkle of magic~")
