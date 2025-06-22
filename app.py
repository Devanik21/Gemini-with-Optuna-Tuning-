import streamlit as st
import google.generativeai as genai
import optuna

# --- Sidebar Configuration ---
st.sidebar.title("⚙️ Configuration")
api_key = st.sidebar.text_input("Google AI API Key", type="password", help="Get your key from https://aistudio.google.com/app/apikey")

# --- Sample Prompt Templates ---
prompt_templates = [
    "Predict the next word: '{}'",
    "What comes next after '{}'",
    "Continue the phrase: '{}'",
    "'{}' then?",
    "Next word after '{}':",
]

# --- Define Objective Function ---
def make_objective(text_input, model):
    def objective(trial):
        template = trial.suggest_categorical("template", prompt_templates)
        full_prompt = template.format(text_input)
        try:
            response = model.generate_content(full_prompt)
            output = response.text.strip().split()
            if len(output) > 0:
                score = len(output[0])  # Shorter next word = better (you can change logic)
            else:
                score = 10
        except Exception:
            score = 10  # Penalize errors
        return score
    return objective

# --- Gemini Setup + UI ---
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        st.session_state.model = model
        st.sidebar.success("✨ Gemini model configured!")
        
        st.title("🔮 Gemini Prompt Optimizer for Next Word Prediction")
        text_input = st.text_input("Enter partial sentence:", "The stars are")

        if st.button("🔍 Tune Best Prompt"):
            with st.spinner("Tuning prompt with Optuna..."):
                objective = make_objective(text_input, model)
                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=10)
                best_template = study.best_params["template"]
                final_prompt = best_template.format(text_input)
                final_output = model.generate_content(final_prompt).text.strip()
                
                st.success(f"✨ Best Prompt: {best_template}")
                st.write("📝 Gemini's next word prediction:")
                st.markdown(f"**{final_output}**")

    except Exception as e:
        st.sidebar.error(f"Invalid API Key: {e}")
else:
    st.warning("Please enter your API key to begin.")

# Footer
st.markdown("---")
st.caption("💖 Prompt tuning with Gemini + Optuna = magic sparkles~")
