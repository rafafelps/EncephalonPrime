CXX       := g++
CXX_FLAGS := 

BIN         := bin
SRC         := src
INCLUDE     := include
LIBS        := 
LIBRARIES   := 
EXECUTABLE  := build


all: $(EXECUTABLE)

run: clean all
	cls
	@echo "🚀 Executing..."
	./$(EXECUTABLE)

$(EXECUTABLE): $(SRC)/*.cpp
	@echo "🚧 Building..."
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) $(LIBS) $^ -o $@ $(LIBRARIES)

clean:
	@echo "🧹 Clearing..."
	-rm $(BIN)/*
