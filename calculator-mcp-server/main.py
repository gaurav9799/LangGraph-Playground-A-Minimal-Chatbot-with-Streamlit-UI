from fastmcp import FastMCP

mcp=FastMCP("Calculator MCP Server")

@mcp.tool
def add(
    first_num:float,
    second_num:float
)->dict:
    """Performs a basic addition operation on two numbers."""
    result=first_num+second_num

    return {
        "first_num":first_num,
        "second_num": second_num,
        "result": result
    }

@mcp.tool
def subtract(
    first_num: float,
    second_num: float
) -> dict:
    """Performs a basic subtraction operation on two numbers."""
    result = first_num - second_num

    return {
        "first_num": first_num,
        "second_num": second_num,
        "result": result
    }

@mcp.tool
def multiply(
    first_num: float,
    second_num: float
) -> dict:
    """Performs a basic multiplication operation on two numbers."""
    result = first_num * second_num

    return {
        "first_num": first_num,
        "second_num": second_num,
        "result": result
    }

@mcp.tool
def divide(
    first_num: float,
    second_num: float
) -> dict:
    """Performs a basic division operation on two numbers."""
    if second_num == 0:
        return {
            "first_num": first_num,
            "second_num": second_num,
            "error": "Division by zero is not allowed"
        }

    result = first_num / second_num

    return {
        "first_num": first_num,
        "second_num": second_num,
        "result": result
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")
