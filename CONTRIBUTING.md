# 🤝 Contributing to Plant Disease Detector

We're thrilled that you're interested in contributing to the Plant Disease Detector project! This AI system is helping farmers worldwide, and every contribution makes a real difference in global food security. 🌾

## 🌟 Ways to Contribute

### 🐛 **Bug Reports**
- Found a bug? Please report it!
- Security vulnerabilities should be reported privately

### 💡 **Feature Requests**
- New disease detection capabilities
- Performance improvements
- User interface enhancements
- Mobile app features

### 📝 **Documentation**
- Improve existing documentation
- Add tutorials and guides
- Translate documentation
- Create video tutorials

### 🧪 **Code Contributions**
- Bug fixes
- New features
- Performance optimizations
- Test coverage improvements

### 📊 **Data Contributions**
- High-quality plant disease images
- Dataset cleaning and validation
- Data augmentation techniques
- New disease categories

## 🚀 Getting Started

### 1. **Fork the Repository**
```bash
# Click the "Fork" button on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Plant_Disease_Detector.git
cd Plant_Disease_Detector
```

### 2. **Set Up Development Environment**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### 3. **Create a Feature Branch**
```bash
# Create and switch to a new branch
git checkout -b feature/amazing-new-feature

# Or for bug fixes
git checkout -b fix/bug-description
```

## 📋 Development Workflow

### **Before You Start**
1. **Check existing issues** to avoid duplicate work
2. **Open an issue** to discuss major changes
3. **Review the project roadmap** in README.md
4. **Read our code style guidelines** below

### **During Development**
1. **Write clear commit messages**
2. **Add tests** for new functionality
3. **Update documentation** as needed
4. **Follow coding standards**
5. **Test your changes** thoroughly

### **Ready to Submit**
1. **Run all tests** and ensure they pass
2. **Check code formatting** with black
3. **Verify lint checks** pass
4. **Update CHANGELOG.md** if applicable
5. **Create a pull request**

## 🧪 Testing Guidelines

### **Running Tests**
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src/ --cov-report=html

# Run specific test file
python -m pytest tests/test_models.py -v

# Run tests for specific function
python -m pytest tests/test_models.py::test_model_training -v
```

### **Writing Tests**
- **Unit tests** for individual functions
- **Integration tests** for workflows
- **API tests** for endpoints
- **Model tests** for accuracy/performance

**Test File Structure:**
```python
import pytest
from src.models.training import create_model

def test_create_model():
    """Test model creation with valid parameters."""
    model = create_model(num_classes=5)
    assert model is not None
    assert len(model.layers) > 0

def test_create_model_invalid_classes():
    """Test model creation with invalid parameters."""
    with pytest.raises(ValueError):
        create_model(num_classes=0)
```

## 🎨 Code Style Guidelines

### **Python Code Standards**
- **PEP 8** compliance
- **Black** for code formatting
- **Flake8** for linting
- **Type hints** where applicable
- **Docstrings** for all functions and classes

### **Code Formatting**
```bash
# Format code with black
black src/ tests/

# Check linting with flake8
flake8 src/ tests/ --max-line-length=88

# Sort imports
isort src/ tests/
```

### **Docstring Format**
```python
def predict_disease(image_path, model_path, confidence_threshold=0.8):
    """
    Predict plant disease from image.
    
    Args:
        image_path (str): Path to the input image file
        model_path (str): Path to the trained model file
        confidence_threshold (float): Minimum confidence for prediction
        
    Returns:
        dict: Prediction results containing:
            - disease (str): Predicted disease name
            - confidence (float): Prediction confidence score
            - top_predictions (list): Top 3 predictions with scores
            
    Raises:
        FileNotFoundError: If image or model file not found
        ValueError: If confidence_threshold not in [0, 1]
        
    Example:
        >>> result = predict_disease("plant.jpg", "model.h5")
        >>> print(result['disease'])
        'Tomato_Early_Blight'
    """
```

### **Commit Message Format**
```
type(scope): brief description

Detailed explanation if needed

Fixes #issue_number
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Maintenance tasks

**Examples:**
```
feat(models): add support for new plant species detection

fix(api): resolve image upload size limitation

docs(readme): update installation instructions

test(models): add unit tests for data preprocessing
```

## 📝 Pull Request Process

### **Before Submitting**
- [ ] **Tests pass** locally
- [ ] **Code is formatted** with black
- [ ] **Documentation updated** if needed
- [ ] **CHANGELOG.md updated** for user-facing changes
- [ ] **Issue linked** if applicable

### **PR Description Template**
```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

## Screenshots (if applicable)
Add screenshots to help explain your changes.

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

Fixes #(issue_number)
```

### **Review Process**
1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainers
3. **Testing** on different environments
4. **Documentation review** if applicable
5. **Approval and merge**

## 🏗️ Development Setup Details



### **Environment Variables**
Create `.env` file for local development:
```bash
# Model paths
MODEL_PATH=models/best_model_checkpoint.h5
DATA_PATH=data/processed/

# API configuration
API_HOST=localhost
API_PORT=5000
DEBUG=True

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

### **Database Setup (if applicable)**
```bash
# For future database features
# Setup instructions will go here
```

## 🎯 Specific Contribution Areas

### **🔬 Model Improvements**
- **Architecture optimization**
- **Hyperparameter tuning**
- **New backbone networks**
- **Ensemble methods**
- **Transfer learning approaches**

**Guidelines:**
- Maintain or improve 81.4% accuracy
- Document performance comparisons
- Include training logs and metrics
- Test on multiple datasets

### **📱 Mobile Development**
- **React Native app**
- **Flutter implementation**
- **Progressive Web App**
- **Offline model support**

### **🌐 Web Interface**
- **Streamlit improvements**
- **React frontend**
- **Vue.js dashboard**
- **Real-time predictions**

### **🔧 Infrastructure**
- **Docker optimization**
- **Kubernetes deployment**
- **CI/CD improvements**
- **Performance monitoring**

### **📊 Data Science**
- **Dataset expansion**
- **Data quality improvements**
- **Augmentation techniques**
- **Bias detection and mitigation**

## 🐛 Bug Reports

### **Before Reporting**
1. **Search existing issues** to avoid duplicates
2. **Update to latest version**
3. **Check documentation** for known issues
4. **Try to reproduce** the bug consistently

### **Bug Report Template**
```markdown
**Bug Description**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
 - OS: [e.g. Ubuntu 20.04, Windows 10, macOS 12]
 - Python Version: [e.g. 3.9.7]
 - TensorFlow Version: [e.g. 2.12.0]
 - Browser (if applicable): [e.g. Chrome 96]

**Additional Context**
Add any other context about the problem here.

**Error Logs**
```
Paste any relevant error logs here
```
```

## 💡 Feature Requests

### **Feature Request Template**
```markdown
**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions.

**Additional context**
Add any other context or screenshots about the feature request.

**Impact Assessment**
- Who would benefit from this feature?
- How critical is this feature?
- Are there any potential downsides?
```

## 🏆 Recognition

### **Contributors Hall of Fame**
We maintain a list of contributors in our README.md and celebrate significant contributions through:

- **GitHub badges** for different contribution types
- **Social media shoutouts** for major features
- **Conference presentation opportunities**
- **Co-authorship** on research papers (when applicable)

### **Contribution Types**
- 🐛 **Bug fixes**
- ⭐ **New features**
- 📝 **Documentation**
- 🧪 **Testing**
- 🎨 **Design**
- 💡 **Ideas & Planning**
- 📊 **Data contributions**
- 🌍 **Translation**

## 📞 Getting Help

### **Communication Channels**
- **GitHub Issues**: Technical questions and bug reports
- **GitHub Discussions**: General questions and feature ideas
- **Email**: [ancur4u@gmail.com] for sensitive issues

### **Response Times**
- **Bug reports**: Within 48 hours
- **Feature requests**: Within 1 week
- **Pull requests**: Within 1 week
- **Security issues**: Within 24 hours

## 📚 Resources

### **Learning Resources**
- [TensorFlow Documentation](https://tensorflow.org/guide)
- [Keras API Reference](https://keras.io/api/)
- [Computer Vision Best Practices](https://github.com/microsoft/computervision-recipes)
- [Agricultural AI Papers](https://paperswithcode.com/task/plant-disease-detection)

### **Development Tools**
- **IDE**: VSCode with Python extension
- **Testing**: pytest, pytest-cov
- **Formatting**: black, isort
- **Linting**: flake8
- **Type Checking**: mypy (optional)

## 🙏 Thank You!

Every contribution, no matter how small, helps improve this project and makes a real difference in global agriculture. Thank you for being part of our mission to use AI for food security! 🌾🤖

---

**Happy Contributing!** 🚀

For questions about contributing, please open an issue or reach out to the maintainers.
