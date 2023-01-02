import jax, gradio

def greet(name):
    seed=sum([ord(n) for n in name ])
    key = jax.random.PRNGKey(seed)
    value=jax.random.maxwell(key).item()
    result=None
    if value>4:
        result="超大吉"
    elif value>3:
        result="大吉"
    elif value>2:
        result="吉"
    elif value>1:
        result="末吉"
    elif value>1:
        result="凶"
    else:
        result="大凶"

    return "再現性おみくじ\n🤖"+name+"の運勢は... " + result + "!"
demo = gradio.Interface(
    fn=greet,
    inputs=gradio.Textbox(lines=2, placeholder="Name Here..."),
    outputs="text",
)
demo.launch()
