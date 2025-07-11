from . import server
import asyncio
import argparse


def main():
    """Main entry point for the package."""
    parser = argparse.ArgumentParser(description="MCP Data Science Server")
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio", "sse", "websocket"],
        help="Transport protocol to use (default: stdio)",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()
    asyncio.run(server.main(transport=args.transport, debug=args.debug))


# Optionally expose other important items at package level
__all__ = ["main", "server"]
