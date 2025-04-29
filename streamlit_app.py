import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chat_models import ChatOpenAI

# Load environment variables (like API keys)
load_dotenv()

# Set up the response schema
response_schemas = [
    ResponseSchema(name="week_name", description="The week name based on cycle"),
    ResponseSchema(name="cycle_day", description="Current cycle day number"),
    ResponseSchema(name="role", description="The role of the user, either 'host' or 'guest'"),
    ResponseSchema(name="pronouns", description="Pronouns used by the user"),
    ResponseSchema(name="host_name", description="Name of host"),
    ResponseSchema(name="Today's Insight", description="A personalized supportive message"),
    ResponseSchema(name="DO", description="Suggested activities to do"),
    ResponseSchema(name="EAT", description="Suggested foods to eat"),
    ResponseSchema(name="MOVE", description="Suggested physical movements"),
    ResponseSchema(name="SEX", description="Suggested intimacy tips")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# Prompt Template
prompt = PromptTemplate(
    template = """
You are a helpful assistant that generates a daily supportive message based on cycle tracking data.

Here are the inputs:
- Cycle Day: {cycle_day}
- Role: {role}
- Week Name: {week_name}
- Hormone Phase: {hormone_phase}
- Hormone Trends: {hormone_trends}
- Emotional & Cognitive States: {emotional_cognitive_states}
- Host Name : {host_name}
- Pronoun: {pronoun}

Instructions:
- If Role is "host", speak directly to the user using their pronouns where needed.
- If Role is "guest", talk about the host using their name and pronouns.
- Create an emotionally supportive, motivational, and natural sounding message.
- Use the Suggested Actions creatively in the output.

{format_instructions}
""",
    input_variables=[
        "cycle_day", 
        "role", 
        "week_name", 
        "hormone_phase", 
        "hormone_trends", 
        "emotional_cognitive_states", 
        "host_name", 
        "pronoun"
    ],
    partial_variables={"format_instructions": format_instructions}
)

# Streamlit UI
st.title("Cycle Supportive Message Generator")

with st.form("input_form"):
    cycle_day = st.number_input("Cycle Day", min_value=1, max_value=50, value=14)
    role = st.selectbox("Role", ["host", "guest"])
    week_name = st.text_input("Week Name", value="Power Week")
    hormone_phase = st.text_input("Hormone Phase", value="Ovulatory")
    hormone_trends = st.text_input("Hormone Trends", value="Estrogen peak")
    emotional_cognitive_states = st.text_area("Emotional & Cognitive States", value="confident, social, energized")
    host_name = st.text_input("Host Name", value="")
    pronoun = st.text_input("Pronouns", value="she/her")
    submit = st.form_submit_button("Generate Message")

if submit:
    input_data = {
        "cycle_day": cycle_day,
        "role": role,
        "week_name": week_name,
        "hormone_phase": hormone_phase,
        "hormone_trends": hormone_trends,
        "emotional_cognitive_states": emotional_cognitive_states,
        "host_name": host_name,
        "pronoun": pronoun
    }

    final_prompt = prompt.format(**input_data)

    # LLM call
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    response = llm.predict(final_prompt)

    # Parse the output
    parsed_response = output_parser.parse(response)

    st.subheader("Generated Supportive Message")
    st.json(parsed_response)
