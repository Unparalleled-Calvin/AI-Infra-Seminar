from torch.utils.cpp_extension import load_inline

cpp_source = """
std::string hello_world() {
  return "Hello World!";
}
"""

module = load_inline(
    name="moss_op",
    cpp_sources=[cpp_source],
    functions=["hello_world"],
    verbose=True,
)

print(module.hello_world())
