default: build

build:
    mkdir -p build && cd build && cmake .. && make && cd .. && cp ./build/alglin_cpp/alglin_cpp.so ./alglin_python/alglin_cpp.so

test:
    python3 -c "import alglin; print(alglin.greet('World'))"

# install:
#     just build
#     pip install -e .

clean:
    rm -rf build
