import streamlit as st
import google.generativeai as genai
import optuna

# --- Sidebar Configuration ---
st.sidebar.title("⚙️ Configuration")
api_key = st.sidebar.text_input(
    "Google AI API Key", type="password",
    help="Get your key from https://aistudio.google.com/app/apikey"
)

# --- About Section ---
with st.sidebar.expander("ℹ️ About"):
    st.markdown(
        """
- **What it does**: Tunes prompt templates to get the best next-word prediction from Gemini.
- **Configuration**: Enter your Gemini API key in the sidebar.
- **Input**: Partial sentence provided by the user.
- **Prompt Templates**: A set of five template formats is tested.
- **Tuning**: Uses Optuna to try different templates and selects the one that yields the shortest next word.
- **Output**: Shows the best template and Gemini's predicted next word.
- **Libraries**: Streamlit for UI, google-generativeai for Gemini, Optuna for hyperparameter search.
        """
    )

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
        # Choose a template and format with user input
        template = trial.suggest_categorical("template", prompt_templates)
        full_prompt = template.format(text_input)
        try:
            response = model.generate_content(full_prompt)
            output = response.text.strip().split()
            # Score: length of the first predicted word (shorter = better)
            score = len(output[0]) if output else 10
        except Exception:
            score = 10  # Penalize any errors
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





# --- About Section ---

# Footer
st.markdown("---")
st.caption("💖 Prompt tuning with Gemini + Optuna = magic sparkles~")
