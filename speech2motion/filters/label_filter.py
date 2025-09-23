import numpy as np

from speech2motion.utils.log import setup_logger

from ..data_structures.motion_record import MotionRecord
from ..index.base_index_mapping import BaseIndexMapping
from ..utils.super import Super
from .base_filter import BaseFilter


def _tokenize_expression(expr: str) -> list[str]:
    """Tokenize expression into tokens (labels, operators, parentheses).

    Args:
        expr (str):
            Expression string to tokenize.

    Returns:
        list[str]:
            List of tokens including labels, operators, and parentheses.

    Raises:
        InvalidCharacterError:
            Raised when expression contains invalid characters.
    """
    tokens = []
    i = 0
    valid_operators = set('()&|!')

    while i < len(expr):
        if expr[i].isspace():
            i += 1
        elif expr[i] in valid_operators:
            tokens.append(expr[i])
            i += 1
        else:
            # read label name
            start = i
            while i < len(expr) and \
                    expr[i] not in valid_operators \
                    and not expr[i].isspace():
                # check if contains invalid characters
                if not (expr[i].isalnum() or expr[i] in '_-'):
                    char = expr[i]
                    msg = (f"Expression contains invalid character: '{char}' "
                           f"(position: {i!s})")
                    raise InvalidCharacterError(msg)
                i += 1

            if start < i:  # ensure we actually read some characters
                tokens.append(expr[start:i])

    return tokens


class ExpressionParser(Super):
    """Recursive descent parser for parsing logical expressions.
    """

    def __init__(
            self,
            tokens: list[str],
            mapping: BaseIndexMapping,
            logger_cfg: None | dict = None) -> None:
        """Initialize expression parser.

        Args:
            tokens (list[str]):
                List of tokens to parse.
            mapping (BaseIndexMapping):
                Index mapping instance for label to ID conversion.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use class name. Defaults to None.
        """
        Super.__init__(self, logger_cfg=logger_cfg)
        self.tokens = tokens
        self.pos = 0
        self.mapping = mapping

    def current_token(self) -> str | None:
        """Get current token without consuming it.

        Returns:
            str | None:
                Current token if available, None if at end of tokens.
        """
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected: str | None = None) -> str | None:
        """Consume current token and advance position.

        Args:
            expected (str | None, optional):
                Expected token to consume. If provided and doesn't match,
                raises appropriate error. Defaults to None.

        Returns:
            str | None:
                Consumed token, None if at end of tokens.

        Raises:
            MismatchedParenthesesError:
                Raised when parentheses don't match.
            InvalidTokenError:
                Raised when token doesn't match expected value.
        """
        if self.pos >= len(self.tokens):
            if expected:
                msg = f"Expected token '{expected}', but reached end of expression"
                if expected == ')':
                    raise MismatchedParenthesesError(msg)
                else:
                    raise InvalidTokenError(msg)
            return None
        token = self.tokens[self.pos]
        self.pos += 1
        if expected and token != expected:
            msg = (f"Expected token '{expected}', but got '{token}' "
                   f"(position: {self.pos-1!s})")
            if expected == ')' or token == '(':
                raise MismatchedParenthesesError(msg)
            else:
                raise InvalidTokenError(msg)
        return token

    async def parse_expression(self, check_remaining: bool = True) -> list | set[int]:
        """Parse expression: or_expr.

        Args:
            check_remaining (bool, optional):
                Whether to check for remaining tokens after parsing.
                Defaults to True.

        Returns:
            list | set[int]:
                Parsed expression result.

        Raises:
            MismatchedParenthesesError:
                Raised when parentheses don't match.
            InvalidTokenError:
                Raised when unexpected tokens are found.
        """
        result = await self.parse_or_expr()

        # only check for remaining tokens at top level
        if check_remaining and self.current_token() is not None:
            remaining_token = self.current_token()
            msg = (f"Unprocessed token after expression parsing: '{remaining_token}' "
                   f"(position: {self.pos!s})")
            if remaining_token == ')':
                raise MismatchedParenthesesError(msg)
            else:
                raise InvalidTokenError(msg)

        return result

    async def parse_or_expr(self) -> list | set[int]:
        """Parse or_expr: and_expr ('|' and_expr)*.

        Returns:
            list | set[int]:
                Parsed OR expression result.
        """
        left = await self.parse_and_expr()

        while self.current_token() == '|':
            self.consume('|')
            right = await self.parse_and_expr()
            left = ['|', left, right]

        return left

    async def parse_and_expr(self) -> list | set[int]:
        """Parse and_expr: not_expr ('&' not_expr)*.

        Returns:
            list | set[int]:
                Parsed AND expression result.
        """
        left = await self.parse_not_expr()

        while self.current_token() == '&':
            self.consume('&')
            right = await self.parse_not_expr()
            left = ['&', left, right]

        return left

    async def parse_not_expr(self) -> list | set[int]:
        """Parse not_expr: '!' primary | primary.

        Returns:
            list | set[int]:
                Parsed NOT expression result.
        """
        if self.current_token() == '!':
            self.consume('!')
            operand = await self.parse_primary()
            return ['!', operand]
        else:
            return await self.parse_primary()

    async def parse_primary(self) -> list | set[int]:
        """Parse primary: '(' expression ')' | label.

        Returns:
            list | set[int]:
                Parsed primary expression result.

        Raises:
            InvalidTokenError:
                Raised when unexpected token is encountered.
            LabelNotFoundError:
                Raised when label is not found in mapping.
        """
        if self.current_token() == '(':
            self.consume('(')
            expr = await self.parse_expression(check_remaining=False)
            # don't check remaining tokens in recursive calls
            self.consume(')')
            return expr
        else:
            # parse label
            label = self.consume()
            if label is None:
                msg = "Expected label or '(', but reached end of expression"
                raise InvalidTokenError(msg)

            # check if label is operator (shouldn't happen, but as extra check)
            if label in '&|!':
                msg = f"Encountered operator '{label}' where label expected"
                raise InvalidTokenError(msg)

            # convert label to corresponding ID set
            try:
                return await self.mapping.get_items(label)
            except KeyError as e:
                msg = f'Label "{label}" does not exist'
                raise LabelNotFoundError(msg) from e


class LabelNotFoundError(Exception):
    """Exception raised when a label is not found in the mapping.
    """
    pass


class InvalidCharacterError(Exception):
    """Exception raised when expression contains invalid characters.
    """
    pass


class MismatchedParenthesesError(Exception):
    """Exception raised when parentheses don't match.
    """
    pass


class EmptyExpressionError(Exception):
    """Exception raised when expression is empty.
    """
    pass


class InvalidTokenError(Exception):
    """Exception raised when an invalid token is encountered.
    """
    pass

class LabelFilter(BaseFilter):
    """Filter for filtering motion_records based on label expressions.
    """

    def __init__(self,
                 name: str,
                 mapping: BaseIndexMapping,
                 logger_cfg: None | dict = None) -> None:
        """Initialize label filter.

        Args:
            name (str):
                Filter name, used as logger name.
            mapping (BaseIndexMapping):
                Index mapping instance that maps labels to a set of IDs.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use class name. Defaults to None.
        """
        BaseFilter.__init__(self, logger_cfg=logger_cfg)
        self.name = name
        self.logger_cfg['logger_name'] = self.name
        self.logger = setup_logger(**self.logger_cfg)
        self.mapping = mapping

    async def reset_mapping(self, mapping: BaseIndexMapping) -> None:
        """Reset index mapping instance.

        Args:
            mapping (BaseIndexMapping):
                New index mapping instance.
        """
        self.mapping = mapping

    async def _not(self, id_set: set[int], universal_set: set[int]) -> set[int]:
        """Negate id_set (complement operation).

        Args:
            id_set (set[int]):
                Set of IDs to negate.
            universal_set (set[int]):
                Universal set for complement operation.

        Returns:
            set[int]:
                Complement of id_set with respect to universal_set.
        """
        return universal_set - id_set

    async def _and(self, id_set0: set[int], id_set1: set[int]) -> set[int]:
        """Intersection of id_set0 and id_set1.

        Args:
            id_set0 (set[int]):
                First set of IDs for intersection.
            id_set1 (set[int]):
                Second set of IDs for intersection.

        Returns:
            set[int]:
                Intersection of the two sets.
        """
        return id_set0.intersection(id_set1)

    async def _or(self, id_set0: set[int], id_set1: set[int]) -> set[int]:
        """Union of id_set0 and id_set1.

        Args:
            id_set0 (set[int]):
                First set of IDs for union.
            id_set1 (set[int]):
                Second set of IDs for union.

        Returns:
            set[int]:
                Union of the two sets.
        """
        return id_set0.union(id_set1)

    async def _calculate(
            self,
            formatted_expression: list[str | set[int]],
            universal_set: set[int]) -> set[int]:
        """Calculate the result of a formatted expression.

        Args:
            formatted_expression (list[str | set[int]]):
                Formatted expression to calculate.
            universal_set (set[int]):
                Universal set for complement operations.

        Returns:
            set[int]:
                Calculated result set of IDs.

        Raises:
            ValueError:
                Raised when unsupported operator is encountered.
        """
        stack = list()
        next_expression = formatted_expression
        while True:
            # nested expressions processed, check if stack has unprocessed expressions
            if isinstance(next_expression, set):
                if len(stack) == 0:
                    break
                else:
                    known_param = next_expression
                    stack_expression = stack.pop()
                    op = stack_expression[0]
                    if op == '!':
                        next_expression = await self._not(known_param, universal_set)
                    elif op == '&':
                        another_param = stack_expression[1]
                        if not isinstance(another_param, set):
                            stack.append([op, known_param])
                            next_expression = another_param
                        else:
                            next_expression = await self._and(
                                known_param, another_param)
                    # op == '|'
                    else:
                        another_param = stack_expression[1]
                        if not isinstance(another_param, set):
                            stack.append([op, known_param])
                            next_expression = another_param
                        else:
                            next_expression = await self._or(known_param, another_param)
            # process expression, if nested, push outer expression to stack
            else:
                op = next_expression[0]
                if op == '!':
                    param = next_expression[1]
                    if isinstance(param, set):
                        next_expression = await self._not(param, universal_set)
                    else:
                        stack.append([op, None])
                        next_expression = param
                elif op == '&':
                    param0 = next_expression[1]
                    param1 = next_expression[2]
                    if not isinstance(param0, set):
                        stack.append([op, param1])
                        next_expression = param0
                    elif not isinstance(param1, set):
                        stack.append([op, param0])
                        next_expression = param1
                    else:
                        next_expression = await self._and(param0, param1)
                elif op == '|':
                    param0 = next_expression[1]
                    param1 = next_expression[2]
                    if not isinstance(param0, set):
                        stack.append([op, param1])
                        next_expression = param0
                    elif not isinstance(param1, set):
                        stack.append([op, param0])
                        next_expression = param1
                    else:
                        next_expression = await self._or(param0, param1)
                else:
                    msg = f'Unsupported operator: {op}'
                    self.logger.error(msg)
                    raise ValueError(msg)
        return next_expression

    async def _parse_expression(
            self,
            expression: str) -> list[str | set[int] | list]:
        """Parse expression and return a formatted list.

        Args:
            expression (str):
                Expression string containing labels and operators (!, &, |, (, )).

        Returns:
            list[str | set[int] | list]:
                Parsed expression that can be used as input for _calculate.
                Each element consists of operator, parameter0, parameter1.
                Parameters can be determined set[int] or nested expressions.

        Raises:
            EmptyExpressionError:
                Raised when expression is empty or contains only whitespace.
            InvalidCharacterError:
                Raised when expression contains invalid characters.
            MismatchedParenthesesError:
                Raised when parentheses don't match.
            InvalidTokenError:
                Raised when invalid token is encountered.
            LabelNotFoundError:
                Raised when label is not found in mapping.
        """
        # check empty string
        if not expression or not expression.strip():
            msg = "Expression is empty or contains only whitespace"
            self.logger.error(msg)
            raise EmptyExpressionError(msg)

        try:
            # tokenize and parse
            tokens = _tokenize_expression(expression)
            if not tokens:
                msg = "No valid tokens after expression parsing"
                self.logger.error(msg)
                raise EmptyExpressionError(msg)

            parser = ExpressionParser(tokens, self.mapping, self.logger_cfg)
            formatted_expression = await parser.parse_expression()

            return formatted_expression

        except (InvalidCharacterError, MismatchedParenthesesError,
                EmptyExpressionError, InvalidTokenError,
                LabelNotFoundError) as e:
            # re-raise our defined exceptions
            msg = f"Error parsing expression '{expression}': {e!s}"
            self.logger.error(msg)
            raise e
        except Exception as e:
            # catch other unexpected exceptions
            msg = f"Unknown error parsing expression '{expression}': {e!s}"
            self.logger.error(msg)
            raise InvalidTokenError(msg) from e

    def _format_expression_for_log(self, expr, indent=0) -> str:
        """Format expression for logging purposes.

        Args:
            expr:
                Expression to format.
            indent (int, optional):
                Indentation level for formatting. Defaults to 0.

        Returns:
            str:
                Formatted expression string for logging.
        """
        indent_str = "  " * indent
        if isinstance(expr, set):
            return f"{indent_str}ID set: {sorted(list(expr))}" \
                if len(expr) < 10 \
                else f"{len(expr)!s} IDs"
        elif isinstance(expr, list) and len(expr) >= 2:
            op = expr[0]
            if op == '!':
                param_str = self._format_expression_for_log(expr[1], indent + 1)
                return f"{indent_str}NOT:\n{param_str}"
            elif op in ['&', '|']:
                op_name = "AND" if op == '&' else "OR"
                param0_str = self._format_expression_for_log(expr[1], indent + 1)
                param1_str = self._format_expression_for_log(expr[2], indent + 1)
                return (f"{indent_str}{op_name}:\n"
                    f"{indent_str}{param0_str}\n"
                    f"{indent_str}{param1_str}")
        return f"{indent_str}Unknown format: {expr}"

    async def filter(self, motion_records: dict[int, MotionRecord],
                     label_expression: str,
               **kwargs) -> dict[int, MotionRecord]:
        """Filter motion_records based on label expression.

        Args:
            motion_records (dict[int, MotionRecord]):
                Motion records to be filtered, key is motion_record id,
                value is motion_record instance.
            label_expression (str):
                Expression string containing labels and operators (!, &, |, (, )).

        Returns:
            dict[int, MotionRecord]:
                Remaining motion_records after filtering, same format as input
                motion_records but possibly reduced in number. Returns empty
                dictionary if no matching motion_record is found.
        """
        if len(motion_records) == 0:
            msg = 'Input motion_records is empty.'
            self.logger.warning(msg)
            return dict()
        universal_set = set(motion_records.keys())
        # parse expression
        formatted_expression = await self._parse_expression(label_expression)
        formatted_expression_str = self._format_expression_for_log(formatted_expression)
        log_message = f"Result of parsing expression '{label_expression}':\n" +\
            f"{formatted_expression_str}"
        self.logger.debug(log_message)
        # calculate qualifying ID set
        available_ids = await self._calculate(formatted_expression, universal_set)

        input_ids = set(motion_records.keys())
        filtered_ids = available_ids.intersection(input_ids)
        ret_dict = dict()
        for k in filtered_ids:
            ret_dict[k] = motion_records[k]
        self.logger.debug(
            f'Out of {len(motion_records)!s} motion records, ' +
            f'after filtering by expression="{label_expression}", ' +
            f'returned {len(ret_dict)!s} motion records.')
        return ret_dict

    async def select_one(self, motion_records: dict[int, MotionRecord],
                         label_expression: str,
                         **kwargs) -> MotionRecord | None:
        """Randomly select one motion record that meets the criteria.

        Args:
            motion_records (dict[int, MotionRecord]):
                Motion records to be selected from, key is motion_record id,
                value is motion_record instance.
            label_expression (str):
                Expression string containing labels and operators (!, &, |, (, )).

        Returns:
            MotionRecord | None:
                Selected motion_record. Returns None if no matching
                motion_record is found.
        """
        if len(motion_records) == 0:
            msg = 'Input motion_records is empty.'
            self.logger.warning(msg)
            return None
        universal_set = set(motion_records.keys())
        # parse expression
        formatted_expression = await self._parse_expression(label_expression)
        # calculate qualifying ID set
        available_ids = await self._calculate(formatted_expression, universal_set)

        input_ids = set(motion_records.keys())
        filtered_ids = available_ids.intersection(input_ids)
        # randomly select one motion record
        if len(filtered_ids) == 0:
            msg = (f'Out of {len(motion_records)!s} motion records, ' +
                   f'after filtering by expression="{label_expression}", ' +
                   'no motion record returned.')
            self.logger.warning(msg)
            return None
        selected_id = np.random.choice(list(filtered_ids))
        self.logger.debug(
            f'Out of {len(motion_records)!s} motion records, ' +
            f'after filtering by expression="{label_expression}", ' +
            f'randomly selected one motion record, id={selected_id!s}.')
        return motion_records[selected_id]
