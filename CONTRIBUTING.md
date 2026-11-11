# Contributing to Wan-Video Pilot

Thank you for contributing to Wan-Video Pilot (`wan-pilot`)! We welcome bug fixes, optimization kernels, additional compiler targets, and community benchmarks.

## Development Workflow

1. **Fork & Clone**:
   ```bash
   git clone https://github.com/EngineerAbdullahBinZafar/wan-pilot.git
   cd wan-pilot
   ```

2. **Virtual Environment Setup**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .
   pip install pytest pytest-mock
   ```

3. **Running the Test Suite**:
   Before submitting any Pull Request, ensure that 100% of the tests pass:
   ```bash
   pytest tests/ -v
   ```

4. **Code Quality Standards**:
   - Maintain documentation integrity and preserve type hints across all module exports.
   - All analytical memory estimations must align with empirical token geometry:
     $(H/16) \times (W/16) \times (1 + (F-1)/4)$.
   - Avoid hardcoded GPU assumptions; provide fallbacks for CPU and Apple Silicon MPS runtimes.

5. **Submitting Pull Requests**:
   - Provide a concise summary of the issue addressed and test coverage added.
   - Keep commits focused and atomic.
