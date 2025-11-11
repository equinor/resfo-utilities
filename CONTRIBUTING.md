# Contributing

The following is a set of guidelines for contributing to resfo-utilities.

## Ground Rules

1. We use pre-commit for checking code style
1. All code must be testable and unit tested.

## Commits

We strive to keep a consistent and clean git history and all contributions should adhere to the following:

1. All tests should pass on all commits(*)
1. A commit should do one atomic change on the repository
1. The commit message should be descriptive.

We expect commit messages to follow the style described [here](https://chris.beams.io/posts/git-commit/). Also, focus on making clear the reasons why you made the change in the first place—the way things worked before the change (and what was wrong with that), the way they work now, and why you decided to solve it the way you did. A commit body is required for anything except very small changes.

() Tip for making sure all tests passes, try out --exec while rebasing. You can then have all tests run per commit in a single command.


## Testing

Tests should run fast, and we currently run all tests in the pre-push
hook, making the need to keep the test runtime low a high priority.

### test naming convention

We prefer a behavioral naming style for tests where each test function starts
with test_that_... instead of just test_....

```python
def test_that_user_cannot_login_with_invalid_password():
    ...
```

This convention comes from the idea of writing tests like executable
specifications, making them read like plain English.

This is to

1. Make test names self-explanatory without reading the body.
1. Encourages a focus on behavior.
1. Easily make changes to the implementation of a behavior, while
   making sure all requirements are still met.

## Type hints

mypy is used to check type hints of all code in src/. This
is important in order to document the public API.

The following guidelines should be applied when adding type hints:

1. Prefer not to use the `Any` type when possible. The `Any` type can
   be convenient, but anything of type `Any` is equivalent to being
   untyped so essentially not type checked.
1. Prefer use of `cast` or `assert` (as a type guard) over using the `#type: ignore`
   to ignore type errors. This is to make the assumption of what types are used
   explicit.

## Docstrings

Avoid adding trivial documentation but where warranted, docstrings should follow the
[google style guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
