#!/bin/bash
# Test runner script for FORGE v1 Neural Audio Workstation
# This script runs different test suites based on provided arguments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Print section
print_section() {
    echo -e "\n${GREEN}▶ $1${NC}\n"
}

# Print warning
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Print error
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Print success
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Function to run tests
run_tests() {
    local test_type=$1
    shift  # Remove first argument
    
    print_section "Running $test_type tests"
    
    if python -m pytest "$@"; then
        print_success "$test_type tests passed!"
        return 0
    else
        print_error "$test_type tests failed!"
        return 1
    fi
}

# Main script
print_header "FORGE v1 Test Suite"

# Check if pytest is installed
if ! python -m pytest --version &> /dev/null; then
    print_error "pytest is not installed. Please run: pip install -r requirements-test.txt"
    exit 1
fi

# Parse command line arguments
case "${1:-all}" in
    "unit")
        print_section "Unit Tests Only"
        run_tests "Unit" tests/unit/ -v -m "not benchmark"
        ;;
    
    "integration")
        print_section "Integration Tests Only"
        run_tests "Integration" tests/integration/test_workflows.py -v -m "not benchmark"
        ;;
    
    "benchmark")
        print_section "Performance Benchmarks"
        print_warning "Benchmark tests may take several minutes to complete..."
        run_tests "Benchmark" tests/ -v -m benchmark
        ;;
    
    "speed")
        print_section "Speed Benchmarks"
        print_warning "Speed tests may take several minutes to complete..."
        run_tests "Speed" tests/integration/test_speed_benchmarks.py -v -m "benchmark and not slow"
        ;;
    
    "fast")
        print_section "Fast Tests (excluding slow benchmarks)"
        run_tests "Fast" tests/ -v -m "not slow and not benchmark"
        ;;
    
    "performance")
        print_section "Performance Module Tests"
        run_tests "Performance" tests/unit/test_performance.py -v
        ;;
    
    "coverage")
        print_section "Test Coverage Report"
        print_warning "This will run all tests and generate coverage reports..."
        python -m pytest tests/ -v -m "not benchmark and not slow" --cov=. --cov-report=term-missing --cov-report=html --cov-report=xml
        print_success "Coverage reports generated:"
        echo "  - Terminal: See above"
        echo "  - HTML: htmlcov/index.html"
        echo "  - XML: coverage.xml"
        ;;
    
    "all")
        print_section "Running All Tests (excluding slow benchmarks)"
        run_tests "All" tests/ -v -m "not slow"
        ;;
    
    "full")
        print_section "Running Full Test Suite (including slow benchmarks)"
        print_warning "This will take several minutes to complete..."
        run_tests "Full" tests/ -v
        ;;
    
    "help"|"-h"|"--help")
        print_header "FORGE v1 Test Runner Help"
        echo ""
        echo "Usage: $0 [test_suite]"
        echo ""
        echo "Available test suites:"
        echo "  unit         - Run only unit tests (fast)"
        echo "  integration  - Run only integration tests"
        echo "  benchmark    - Run performance benchmarks"
        echo "  speed        - Run speed benchmarks only"
        echo "  fast         - Run all tests except slow benchmarks (default for CI)"
        echo "  performance  - Run performance module tests only"
        echo "  coverage     - Run tests with coverage report"
        echo "  all          - Run all tests except slow benchmarks (default)"
        echo "  full         - Run complete test suite including slow benchmarks"
        echo "  help         - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 unit          # Run unit tests only"
        echo "  $0 benchmark     # Run performance benchmarks"
        echo "  $0 coverage      # Generate coverage report"
        echo ""
        ;;
    
    *)
        print_error "Unknown test suite: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac

exit_code=$?

if [ $exit_code -eq 0 ]; then
    print_header "Test Suite Complete ✅"
else
    print_header "Test Suite Failed ❌"
fi

exit $exit_code
