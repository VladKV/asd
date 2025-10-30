"""
Simplified VM code which works for some cases.
You need extend/rewrite code to pass all cases.
"""

import builtins
import dis
import types
import typing as tp


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.12/Include/internal/pycore_frame.h

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members
    """
    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: dict[str, tp.Any],
                 frame_globals: dict[str, tp.Any],
                 frame_locals: dict[str, tp.Any]) -> None:
        self.code = frame_code
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.data_stack: tp.Any = []
        self.return_value = None
        self.pending_kw_names: tuple[str, ...] | None = None

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        return self.data_stack.pop()

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def popn(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def run(self) -> tp.Any:
        # Собираем все инструкции в список и создаем словарь по offset
        instructions_list = list(dis.get_instructions(self.code))
        instruction_map = {inst.offset: inst for inst in instructions_list}
        sorted_offsets = sorted(instruction_map.keys())
        
        # Указатель на текущую инструкцию
        instruction_pointer = 0
        
        while instruction_pointer in instruction_map:
            instruction = instruction_map[instruction_pointer]
            
            # Вызываем обработчик
            handler = getattr(self, instruction.opname.lower() + "_op", None)
            if handler is not None:
                # Всегда передаем полную инструкцию
                result = handler(instruction)
                
                # Если обработчик вернул offset, переходим туда
                if result is not None and isinstance(result, int):
                    instruction_pointer = result
                    continue
            
            # По умолчанию переходим к следующей инструкции
            # Находим следующий offset в списке
            current_idx = sorted_offsets.index(instruction.offset)
            if current_idx < len(sorted_offsets) - 1:
                instruction_pointer = sorted_offsets[current_idx + 1]
            else:
                break  # Это последняя инструкция
        
        return self.return_value
    
    def resume_op(self, instruction: dis.Instruction) -> tp.Any:
        return None

    def push_null_op(self, instruction: dis.Instruction) -> tp.Any:
        self.push(None)

    def precall_op(self, instruction: dis.Instruction) -> tp.Any:
        # Simplified: ignore PRECALL setup
        return None

    def call_op(self, instruction: dis.Instruction) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-CALL
        """
        # Handle possible KW_NAMES provided by preceding KW_NAMES opcode
        kw_names: tuple[str, ...] | None = self.pending_kw_names
        self.pending_kw_names = None

        total_values = instruction.arg
        num_kwargs = len(kw_names) if kw_names else 0
        pos_args_count = total_values - num_kwargs
        stack_len = len(self.data_stack)
        # Extract values destined to arguments (positional first, then keyword values)
        values_slice = self.data_stack[stack_len - total_values:stack_len] if total_values > 0 else []
        # Find callable scanning left from before the values slice
        search_end = stack_len - total_values
        func_index = None
        for i in range(search_end - 1, -1, -1):
            candidate = self.data_stack[i]
            if callable(candidate):
                func_index = i
                break
        if func_index is None:
            raise TypeError("CALL: callable not found on stack")
        null_index = func_index - 1
        has_null = null_index >= 0 and self.data_stack[null_index] is None
        f = self.data_stack[func_index]
        # Split positional and keyword args
        kwargs: dict[str, tp.Any] = {}
        if kw_names:
            arguments = values_slice[:pos_args_count]
            kw_values = values_slice[pos_args_count:]
            kwargs = {name: value for name, value in zip(kw_names, kw_values)}
        else:
            arguments = values_slice
        # Remove consumed stack slice (func, optional NULL, and arg values)
        start_delete = null_index if has_null else func_index
        del self.data_stack[start_delete:]
        self.push(f(*arguments, **kwargs))

    def load_name_op(self, instruction: dis.Instruction) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-LOAD_NAME
        """
        # Resolve name from instruction
        name = instruction.argval
        # Check locals first, then globals, then builtins
        if name in self.locals:
            self.push(self.locals[name])
        elif name in self.globals:
            self.push(self.globals[name])
        elif name in self.builtins:
            self.push(self.builtins[name])
        else:
            raise NameError(f"name '{name}' is not defined")

    def load_global_op(self, instruction: dis.Instruction) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-LOAD_GLOBAL
        """
        # Resolve name from instruction
        name = instruction.argval
        # Check globals, then builtins
        if name in self.globals:
            self.push(self.globals[name])
        elif name in self.builtins:
            self.push(self.builtins[name])
        else:
            raise NameError(f"name '{name}' is not defined")

    def load_const_op(self, instruction: dis.Instruction) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-LOAD_CONST
        """
        self.push(instruction.argval)

    def return_value_op(self, instruction: dis.Instruction) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-RETURN_VALUE
        """
        self.return_value = self.pop()

    def pop_top_op(self, instruction: dis.Instruction) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-POP_TOP
        """
        self.pop()

    def store_name_op(self, instruction: dis.Instruction) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-STORE_NAME
        """
        const = self.pop()
        self.locals[instruction.argval] = const
    
    # Jump operations
    def jump_forward_op(self, instruction: dis.Instruction) -> int:
        """JUMP_FORWARD: jump by relative offset forward"""
        return instruction.offset + instruction.arg
    
    def jump_backward_op(self, instruction: dis.Instruction) -> int:
        """JUMP_BACKWARD: jump by relative offset backward"""
        return instruction.offset - instruction.arg
    
    def pop_jump_if_false_op(self, instruction: dis.Instruction) -> int | None:
        """POP_JUMP_FORWARD_IF_FALSE: pop value, jump if false"""
        value = self.pop()
        if not bool(value):
            return instruction.offset + instruction.arg
        return None
    
    def pop_jump_if_true_op(self, instruction: dis.Instruction) -> int | None:
        """POP_JUMP_FORWARD_IF_TRUE: pop value, jump if true"""
        value = self.pop()
        if bool(value):
            return instruction.offset + instruction.arg
        return None
    
    def pop_jump_backward_if_false_op(self, instruction: dis.Instruction) -> int | None:
        """POP_JUMP_BACKWARD_IF_FALSE"""
        value = self.pop()
        if not bool(value):
            return instruction.offset - instruction.arg
        return None
    
    def pop_jump_backward_if_true_op(self, instruction: dis.Instruction) -> int | None:
        """POP_JUMP_BACKWARD_IF_TRUE"""
        value = self.pop()
        if bool(value):
            return instruction.offset - instruction.arg
        return None
    
    def jump_if_false_or_pop_op(self, instruction: dis.Instruction) -> int | None:
        """JUMP_IF_FALSE_OR_POP: used for 'and' short-circuit"""
        value = self.top()  # Peek without popping
        if not bool(value):
            return instruction.offset + instruction.arg
        else:
            self.pop()  # Pop if true
            return None
    
    def jump_if_true_or_pop_op(self, instruction: dis.Instruction) -> int | None:
        """JUMP_IF_TRUE_OR_POP: used for 'or' short-circuit"""
        value = self.top()  # Peek without popping
        if bool(value):
            return instruction.offset + instruction.arg
        else:
            self.pop()  # Pop if false
            return None
    
    # Binary operations
    def binary_op_op(self, instruction: dis.Instruction) -> None:
        """
        BINARY_OP: binary operator with arg indicating operation type
        In Python 3.12+, this replaces old BINARY_ADD, BINARY_SUBTRACT, etc.
        """
        right = self.pop()
        left = self.pop()
        
        # arg contains operation code; map dynamically using dis._nb_ops
        opcode = instruction.arg
        # Resolve operation name from dis._nb_ops which may be a dict or a list depending on Python version
        op_name: tp.Any = None
        try:
            nb_ops = dis._nb_ops  # type: ignore[attr-defined]
        except Exception:
            nb_ops = None
        if isinstance(nb_ops, dict):
            op_by_code = {code: name for name, code in nb_ops.items()}
            op_name = op_by_code.get(opcode)
        elif isinstance(nb_ops, (list, tuple)):
            if 0 <= opcode < len(nb_ops):
                op_name = nb_ops[opcode]  # type: ignore[index]

        # Normalize op key: accept names like 'ADD', 'NB_POWER' or symbols like '+' '**'
        op_key: str | None = None
        if isinstance(op_name, str):
            op_key = op_name
        elif isinstance(op_name, (list, tuple)):
            # Prefer symbolic operator if present, else the first string token
            symbols = {"+","-","*","/","//","%","**","<<",">>","&","|","^"}
            strings = [s for s in op_name if isinstance(s, str)]
            sym = next((s for s in strings if s in symbols), None)
            op_key = sym or (strings[0] if strings else None)

        if op_key and op_key.startswith("NB_"):
            op_key = op_key[3:]
        # Treat in-place ops the same as their regular counterparts in this VM
        if op_key and op_key.startswith("INPLACE_"):
            op_key = op_key[len("INPLACE_"):]
        # Convert augmented assignment symbols like "+=", "-=", "/=", etc. to their base operator
        if op_key in ('+=', '-=', '*=', '/=', '//=', '%=', '**=', '<<=', '>>=', '&=', '|=', '^='):
            op_key = op_key[:-1]
        
        def true_div(a: tp.Any, b: tp.Any) -> tp.Any:
            return a / b

        if op_key in ('ADD', '+'):
            self.push(left + right)
        elif op_key in ('SUBTRACT', '-'):
            self.push(left - right)
        elif op_key in ('MULTIPLY', '*'):
            self.push(left * right)
        elif op_key in ('TRUE_DIVIDE', 'DIVIDE', '/'):
            self.push(true_div(left, right))
        elif op_key in ('FLOOR_DIVIDE', '//'):
            self.push(left // right)
        elif op_key in ('MODULO', '%'):
            self.push(left % right)
        elif op_key in ('POWER', '**'):
            self.push(left ** right)
        elif op_key in ('LSHIFT', '<<'):
            self.push(left << right)
        elif op_key in ('RSHIFT', '>>'):
            self.push(left >> right)
        elif op_key in ('AND', '&'):
            self.push(left & right)
        elif op_key in ('OR', '|'):
            self.push(left | right)
        elif op_key in ('XOR', '^'):
            self.push(left ^ right)
        else:
            raise ValueError(f"Unknown binary operation code: {opcode} ({op_name})")
    
    # Comparison operations
    def compare_op_op(self, instruction: dis.Instruction) -> None:
        """
        COMPARE_OP: comparison operation with arg indicating comparison type
        """
        right = self.pop()
        left = self.pop()
        
        # Prefer instruction.argval which should be a string like '==', '<', etc.
        name = instruction.argval if isinstance(instruction.argval, str) else None
        if name is None:
            opcode = instruction.arg
            cmp_names = list(dis.cmp_op)
            if 0 <= opcode < len(cmp_names):
                name = cmp_names[opcode]
            else:
                # Fallback for newer enum-like encodings; handle common ones
                mapping: dict[int, str] = {
                    0: '<', 1: '<=', 2: '==', 3: '!=', 4: '>', 5: '>=',
                }
                if opcode in mapping:
                    name = mapping[opcode]
                else:
                    raise ValueError(f"Unknown comparison operation: {opcode}")
        if name == '<':
            self.push(left < right)
        elif name == '<=':
            self.push(left <= right)
        elif name == '==':
            self.push(left == right)
        elif name == '!=':
            self.push(left != right)
        elif name == '>':
            self.push(left > right)
        elif name == '>=':
            self.push(left >= right)
        elif name == 'in':
            self.push(left in right)
        elif name == 'not in':
            self.push(left not in right)
        elif name == 'is':
            self.push(left is right)
        elif name == 'is not':
            self.push(left is not right)
        else:
            raise ValueError(f"Unsupported comparison operation: {name}")

    def is_op_op(self, instruction: dis.Instruction) -> None:
        """IS_OP: identity test 'is' or 'is not'"""
        right = self.pop()
        left = self.pop()
        if instruction.arg == 0:  # is
            self.push(left is right)
        elif instruction.arg == 1:  # is not
            self.push(left is not right)
        else:
            raise ValueError(f"Unsupported IS_OP arg: {instruction.arg}")

    def contains_op_op(self, instruction: dis.Instruction) -> None:
        """CONTAINS_OP: membership test 'in' or 'not in'"""
        right = self.pop()
        left = self.pop()
        if instruction.arg == 0:  # in
            self.push(left in right)
        elif instruction.arg == 1:  # not in
            self.push(left not in right)
        else:
            raise ValueError(f"Unsupported CONTAINS_OP arg: {instruction.arg}")
    
    # Unary operations
    def unary_negative_op(self, instruction: dis.Instruction) -> None:
        """UNARY_NEGATIVE: -a"""
        value = self.pop()
        self.push(-value)
    
    def unary_positive_op(self, instruction: dis.Instruction) -> None:
        """UNARY_POSITIVE: +a"""
        value = self.pop()
        self.push(+value)
    
    def unary_not_op(self, instruction: dis.Instruction) -> None:
        """UNARY_NOT: not a"""
        value = self.pop()
        self.push(not value)
    
    def unary_invert_op(self, instruction: dis.Instruction) -> None:
        """UNARY_INVERT: ~a"""
        value = self.pop()
        self.push(~value)
    
    # Fast/local operations
    def load_fast_op(self, instruction: dis.Instruction) -> None:
        """LOAD_FAST: load local variable by index"""
        var_name = self.code.co_varnames[instruction.arg]
        if var_name in self.locals:
            self.push(self.locals[var_name])
        else:
            raise NameError(f"name '{var_name}' is not defined")
    
    def store_fast_op(self, instruction: dis.Instruction) -> None:
        """STORE_FAST: store into local variable by index"""
        var_name = self.code.co_varnames[instruction.arg]
        value = self.pop()
        self.locals[var_name] = value
    
    def delete_fast_op(self, instruction: dis.Instruction) -> None:
        """DELETE_FAST: delete local variable"""
        var_name = self.code.co_varnames[instruction.arg]
        if var_name in self.locals:
            del self.locals[var_name]
        else:
            raise NameError(f"name '{var_name}' is not defined")
    
    # Subscription operations
    def binary_subscr_op(self, instruction: dis.Instruction) -> None:
        """BINARY_SUBSCR: obj[index]"""
        index = self.pop()
        obj = self.pop()
        self.push(obj[index])
    
    def store_subscr_op(self, instruction: dis.Instruction) -> None:
        """STORE_SUBSCR: obj[index] = value"""
        value = self.pop()
        index = self.pop()
        obj = self.pop()
        obj[index] = value
    
    def delete_subscr_op(self, instruction: dis.Instruction) -> None:
        """DELETE_SUBSCR: del obj[index]"""
        index = self.pop()
        obj = self.pop()
        del obj[index]
    
    # Container building
    def build_list_op(self, instruction: dis.Instruction) -> None:
        """BUILD_LIST: build list from stack"""
        count = instruction.arg
        items = self.popn(count)
        self.push(items)
    
    def build_tuple_op(self, instruction: dis.Instruction) -> None:
        """BUILD_TUPLE: build tuple from stack"""
        count = instruction.arg
        items = self.popn(count)
        self.push(tuple(items))
    
    def build_set_op(self, instruction: dis.Instruction) -> None:
        """BUILD_SET: build set from stack"""
        count = instruction.arg
        items = self.popn(count)
        self.push(set(items))
    
    def build_map_op(self, instruction: dis.Instruction) -> None:
        """BUILD_MAP: build dict from stack"""
        count = instruction.arg
        items = self.popn(count * 2)  # key-value pairs
        result = {}
        for i in range(0, len(items), 2):
            result[items[i]] = items[i + 1]
        self.push(result)
    
    def build_const_key_map_op(self, instruction: dis.Instruction) -> None:
        """BUILD_CONST_KEY_MAP: build dict with const keys"""
        keys_tuple = self.code.co_consts[instruction.arg]
        num_values = len(keys_tuple)
        values = self.popn(num_values)
        result = dict(zip(keys_tuple, values))
        self.push(result)
    
    # Unpacking
    def unpack_sequence_op(self, instruction: dis.Instruction) -> None:
        """UNPACK_SEQUENCE: unpack sequence into individual items"""
        count = instruction.arg
        seq = self.pop()
        items = list(seq)
        if len(items) != count:
            raise ValueError(f"not enough values to unpack (expected {count}, got {len(items)})")
        # Push in reverse order so first items are on top
        for item in reversed(items):
            self.push(item)
    
    def unpack_ex_op(self, instruction: dis.Instruction) -> None:
        """UNPACK_EX: unpack with *varargs"""
        count = instruction.arg  # number of items after varargs
        seq = self.pop()
        items = list(seq)
        varargs_count = (len(items) - count) if len(items) > count else 0
        
        # Push in reverse: varargs first, then individual items
        for item in reversed(items[:count]):
            self.push(item)
        self.push(items[count:])  # varargs list
    
    # Iteration
    def get_iter_op(self, instruction: dis.Instruction) -> None:
        """GET_ITER: get iterator from iterable"""
        obj = self.pop()
        self.push(iter(obj))
    
    def for_iter_op(self, instruction: dis.Instruction) -> int | None:
        """FOR_ITER: get next item from iterator or jump if exhausted"""
        iterator = self.data_stack[-1]  # Peek at top without popping
        try:
            value = next(iterator)
            self.push(value)
            return None  # Continue to next instruction
        except StopIteration:
            # Jump forward to end of loop
            return instruction.offset + instruction.arg
    
    def end_for_op(self, instruction: dis.Instruction) -> None:
        """END_FOR: pop iterator from stack"""
        self.pop()  # Remove iterator
    
    # Global operations
    def store_global_op(self, instruction: dis.Instruction) -> None:
        """STORE_GLOBAL: store into global variable"""
        value = self.pop()
        self.globals[instruction.argval] = value
    
    def delete_global_op(self, instruction: dis.Instruction) -> None:
        """DELETE_GLOBAL: delete global variable"""
        if instruction.argval in self.globals:
            del self.globals[instruction.argval]
        else:
            raise NameError(f"name '{instruction.argval}' is not defined")
    
    def delete_name_op(self, instruction: dis.Instruction) -> None:
        """DELETE_NAME: delete name from current scope"""
        if instruction.argval in self.locals:
            del self.locals[instruction.argval]
        elif instruction.argval in self.globals:
            del self.globals[instruction.argval]
        else:
            raise NameError(f"name '{instruction.argval}' is not defined")
    
    # Attribute operations
    def load_attr_op(self, instruction: dis.Instruction) -> None:
        """LOAD_ATTR: load attribute from object"""
        obj = self.pop()
        attr_name = instruction.argval
        value = getattr(obj, attr_name)
        self.push(value)
    
    def store_attr_op(self, instruction: dis.Instruction) -> None:
        """STORE_ATTR: store attribute to object"""
        value = self.pop()
        obj = self.pop()
        attr_name = instruction.argval
        setattr(obj, attr_name, value)
    
    def delete_attr_op(self, instruction: dis.Instruction) -> None:
        """DELETE_ATTR: delete attribute from object"""
        obj = self.pop()
        attr_name = instruction.argval
        delattr(obj, attr_name)
    
    # Slice operations
    def build_slice_op(self, instruction: dis.Instruction) -> None:
        """BUILD_SLICE: build slice object"""
        if instruction.arg == 2:
            stop = self.pop()
            start = self.pop()
            self.push(slice(start, stop))
        elif instruction.arg == 3:
            step = self.pop()
            stop = self.pop()
            start = self.pop()
            self.push(slice(start, stop, step))
        else:
            raise ValueError(f"Invalid slice arg count: {instruction.arg}")
    
    def binary_slice_op(self, instruction: dis.Instruction) -> None:
        """BINARY_SLICE: obj[lower:upper]"""
        slice_obj = self.pop()
        obj = self.pop()
        self.push(obj[slice_obj])
    
    def store_slice_op(self, instruction: dis.Instruction) -> None:
        """STORE_SLICE: obj[lower:upper] = value"""
        value = self.pop()
        slice_obj = self.pop()
        obj = self.pop()
        obj[slice_obj] = value
    
    def delete_slice_op(self, instruction: dis.Instruction) -> None:
        """DELETE_SLICE: del obj[lower:upper]"""
        slice_obj = self.pop()
        obj = self.pop()
        del obj[slice_obj]
    
    # String operations
    def build_string_op(self, instruction: dis.Instruction) -> None:
        """BUILD_STRING: build string from multiple items on stack"""
        count = instruction.arg
        items = self.popn(count)
        self.push(''.join(str(item) for item in items))

    # Keyword names marker for CALL
    def kw_names_op(self, instruction: dis.Instruction) -> None:
        """KW_NAMES: record tuple of keyword argument names for next CALL"""
        # instruction.argval is the tuple of keyword argument names
        self.pending_kw_names = instruction.argval
    
    # NOP
    def nop_op(self, instruction: dis.Instruction) -> None:
        """NOP: no operation"""
        pass
    
    # Copy operations
    def copy_op(self, instruction: dis.Instruction) -> None:
        """COPY: push a copy of stack item at depth arg (0 = TOS)"""
        depth = instruction.arg
        if depth < 0:
            raise IndexError("COPY: invalid stack depth")
        # Clamp depth to the deepest available item
        if depth >= len(self.data_stack):
            depth = len(self.data_stack) - 1
        self.push(self.data_stack[-(depth + 1)])
    
    # Function operations - improved
    def make_function_op(self, instruction: dis.Instruction) -> None:
        """MAKE_FUNCTION: create function from code and qualname"""
        code = self.pop()  # the code associated with the function
        
        # Parse flags from instruction.arg
        flags = instruction.arg
        
        # Get qualname for the function
        qualname = self.pop() if flags & 0x08 else code.co_name
        
        # Get defaults, annotations if present
        num_defaults = 0
        defaults = ()
        if flags & 0x01:  # positional arguments default
            defaults_tuple = self.pop()
            defaults = tuple(defaults_tuple) if hasattr(defaults_tuple, '__iter__') else (defaults_tuple,)
            num_defaults = len(defaults)
        
        kwdefaults = {}
        if flags & 0x02:  # keyword arguments default
            kwdefaults = self.pop()
        
        annotations = {}
        if flags & 0x04:  # annotations
            annotations = self.pop()
        
        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            # Create locals dict
            f_locals: dict[str, tp.Any] = {}
            
            # Process positional arguments
            pos_args = list(args[:code.co_argcount])
            for i, name in enumerate(code.co_varnames[:code.co_argcount]):
                if i < len(pos_args):
                    f_locals[name] = pos_args[i]
                elif i >= code.co_argcount - num_defaults:
                    f_locals[name] = defaults[i - (code.co_argcount - num_defaults)]
                else:
                    raise TypeError(f"{code.co_name}() missing required positional argument: '{name}'")
            
            # Process *args
            if code.co_flags & 0x04:  # CO_VARARGS
                varargs_start = code.co_argcount
                f_locals[code.co_varnames[varargs_start]] = tuple(args[code.co_argcount:])
            
            # Process **kwargs
            if code.co_flags & 0x08:  # CO_VARKW
                varkw_start = code.co_argcount + (1 if code.co_flags & 0x04 else 0)
                f_locals[code.co_varnames[varkw_start]] = kwargs
            
            # Process keyword arguments from kwargs
            for name, value in kwargs.items():
                if name in f_locals:
                    f_locals[name] = value
                elif name not in code.co_varnames:
                    raise TypeError(f"{code.co_name}() got an unexpected keyword argument '{name}'")
            
            # Create frame with function's globals
            frame = Frame(code, self.builtins, self.globals, f_locals)
            return frame.run()
        
        # Set function attributes (do not replace __code__ to avoid free var mismatch)
        f.__qualname__ = qualname
        f.__defaults__ = defaults if defaults else None
        f.__kwdefaults__ = kwdefaults if kwdefaults else None
        f.__annotations__ = annotations
        
        self.push(f)
    
    def call_function_ex_op(self, instruction: dis.Instruction) -> None:
        """CALL_FUNCTION_EX: call function with unpacked arguments"""
        kwargs_dict = self.pop() if instruction.arg & 0x01 else {}
        args_tuple = self.pop()
        func = self.pop()
        
        # Call function with unpacked args
        result = func(*args_tuple, **kwargs_dict)
        self.push(result)


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'], globals_context, globals_context)
        return frame.run()
