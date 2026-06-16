def pytest_addoption(parser):
    # Add a custom command-line flag
    parser.addoption(
        "--record",
        action="store_true",
        help="Make live requests to ingrid and save the responses to replay later"
    )
