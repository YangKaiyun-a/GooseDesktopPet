using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace GooseDesktopPet;

public partial class MainWindow : Window
{
    private const string IdleState = "idle";
    private const string JumpState = "action_2";
    private const double DisplayScale = 0.2;
    private const double JumpDistance = 50;
    private const double JumpHeight = 20;
    private const int JumpStartFrameIndex = 4;
    private const int JumpLandingFrameIndex = 11;
    private const double DragThreshold = 5;

    private readonly DispatcherTimer _animationTimer = new();
    private readonly List<string> _stateOrder = [];
    private readonly Dictionary<string, List<BitmapImage>> _framesByState = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _preserveYStates = new(StringComparer.OrdinalIgnoreCase);

    private string _currentState = IdleState;
    private int _currentFrameIndex;
    private bool _isMouseDown;
    private bool _isDragging;
    private Point _mouseDownScreenPoint;
    private Point _windowDownPoint;
    private Point _jumpStartPoint;
    private bool _stateMotionCompleted;

    public MainWindow()
    {
        InitializeComponent();
        Loaded += MainWindow_Loaded;
        _animationTimer.Tick += AnimationTimer_Tick;
    }

    private void MainWindow_Loaded(object sender, RoutedEventArgs e)
    {
        LoadPet(Path.Combine(AppContext.BaseDirectory, "Assets", "Pet", "manifest.json"));
        ConfigureWindowSize();
        Left = SystemParameters.WorkArea.Right - Width - 60;
        Top = SystemParameters.WorkArea.Bottom - Height - 40;
        SwitchState(IdleState);
        _animationTimer.Start();
    }

    private void LoadPet(string manifestPath)
    {
        var manifest = PetManifest.Load(manifestPath);
        _preserveYStates.Clear();
        foreach (var state in manifest.PreserveYStates)
        {
            _preserveYStates.Add(state);
        }

        _stateOrder.Clear();
        _framesByState.Clear();
        var root = Path.GetDirectoryName(manifestPath) ?? AppContext.BaseDirectory;
        foreach (var state in manifest.States)
        {
            if (string.IsNullOrWhiteSpace(state.State) || state.Frames.Count == 0)
            {
                continue;
            }

            _stateOrder.Add(state.State);
            _framesByState[state.State] = state.Frames
                .Select(frame => LoadBitmap(ResolvePath(root, frame)))
                .ToList();
        }

        if (!_framesByState.ContainsKey(IdleState))
        {
            throw new InvalidOperationException("Pet manifest must include an idle state.");
        }

        var fps = manifest.TargetFps <= 0 ? 12 : manifest.TargetFps;
        _animationTimer.Interval = TimeSpan.FromMilliseconds(1000 / fps);
    }

    private void ConfigureWindowSize()
    {
        var firstFrame = _framesByState[IdleState][0];
        Width = Math.Max(1, firstFrame.PixelWidth * DisplayScale);
        Height = Math.Max(1, firstFrame.PixelHeight * DisplayScale);
    }

    private static string ResolvePath(string root, string path)
    {
        if (Path.IsPathRooted(path))
        {
            var assetIndex = path.IndexOf("/Assets/Pet/", StringComparison.OrdinalIgnoreCase);
            if (assetIndex >= 0)
            {
                var relativeAsset = path[(assetIndex + 1)..].Replace('/', Path.DirectorySeparatorChar);
                return Path.Combine(AppContext.BaseDirectory, relativeAsset);
            }

            var stateIndex = path.IndexOf("/states/", StringComparison.OrdinalIgnoreCase);
            if (stateIndex >= 0)
            {
                var relativeStatePath = path[(stateIndex + 1)..].Replace('/', Path.DirectorySeparatorChar);
                return Path.Combine(root, relativeStatePath);
            }

            return path;
        }

        return Path.Combine(root, path.Replace('/', Path.DirectorySeparatorChar));
    }

    private static BitmapImage LoadBitmap(string path)
    {
        var image = new BitmapImage();
        image.BeginInit();
        image.CacheOption = BitmapCacheOption.OnLoad;
        image.CreateOptions = BitmapCreateOptions.PreservePixelFormat;
        image.UriSource = new Uri(path, UriKind.Absolute);
        image.EndInit();
        image.Freeze();
        return image;
    }

    private void AnimationTimer_Tick(object? sender, EventArgs e)
    {
        var frames = _framesByState[_currentState];
        if (frames.Count == 0)
        {
            return;
        }

        ResetStateMotionForLoop();
        PetImage.Source = frames[_currentFrameIndex];
        ApplyStateMotion(frames.Count);
        _currentFrameIndex = (_currentFrameIndex + 1) % frames.Count;
    }

    private void ResetStateMotionForLoop()
    {
        if (!string.Equals(_currentState, JumpState, StringComparison.OrdinalIgnoreCase))
        {
            return;
        }

        if (_currentFrameIndex != 0 || !_stateMotionCompleted)
        {
            return;
        }

        _jumpStartPoint = new Point(Left, Top);
        _stateMotionCompleted = false;
    }

    private void ApplyStateMotion(int frameCount)
    {
        if (!string.Equals(_currentState, JumpState, StringComparison.OrdinalIgnoreCase))
        {
            return;
        }

        if (_stateMotionCompleted)
        {
            return;
        }

        if (_currentFrameIndex < JumpStartFrameIndex)
        {
            return;
        }

        var jumpFrameSpan = Math.Max(1, Math.Min(JumpLandingFrameIndex, frameCount - 1) - JumpStartFrameIndex);
        var activeFrameIndex = Math.Clamp(_currentFrameIndex - JumpStartFrameIndex, 0, jumpFrameSpan);
        var progress = activeFrameIndex / (double)jumpFrameSpan;
        var x = -JumpDistance * progress;
        var y = -Math.Sin(progress * Math.PI) * JumpHeight;
        Left = _jumpStartPoint.X + x;
        Top = _jumpStartPoint.Y + y;

        if (_currentFrameIndex >= JumpLandingFrameIndex)
        {
            Left = _jumpStartPoint.X - JumpDistance;
            Top = _jumpStartPoint.Y;
            _jumpStartPoint = new Point(Left, Top);
            _stateMotionCompleted = true;
        }
    }

    private void SwitchState(string state)
    {
        if (!_framesByState.ContainsKey(state))
        {
            state = IdleState;
        }

        _currentState = state;
        _currentFrameIndex = 0;
        _jumpStartPoint = new Point(Left, Top);
        _stateMotionCompleted = false;
        PetImage.Source = _framesByState[_currentState][0];
    }

    private void SwitchToNextState()
    {
        var currentIndex = _stateOrder.FindIndex(state => string.Equals(state, _currentState, StringComparison.OrdinalIgnoreCase));
        var nextIndex = currentIndex < 0 ? 0 : currentIndex + 1;
        if (nextIndex >= _stateOrder.Count)
        {
            SwitchState(IdleState);
            return;
        }

        SwitchState(_stateOrder[nextIndex]);
    }

    private void PetSurface_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        _isMouseDown = true;
        _isDragging = false;
        _mouseDownScreenPoint = PointToScreen(e.GetPosition(this));
        _windowDownPoint = new Point(Left, Top);
        PetSurface.CaptureMouse();
        e.Handled = true;
    }

    private void PetSurface_MouseMove(object sender, MouseEventArgs e)
    {
        if (!_isMouseDown)
        {
            return;
        }

        var currentScreenPoint = PointToScreen(e.GetPosition(this));
        var delta = currentScreenPoint - _mouseDownScreenPoint;
        if (!_isDragging && (Math.Abs(delta.X) > DragThreshold || Math.Abs(delta.Y) > DragThreshold))
        {
            _isDragging = true;
        }

        if (!_isDragging)
        {
            return;
        }

        Left = _windowDownPoint.X + delta.X;
        Top = _windowDownPoint.Y + delta.Y;
        if (string.Equals(_currentState, JumpState, StringComparison.OrdinalIgnoreCase))
        {
            _jumpStartPoint = new Point(Left, Top);
        }

        e.Handled = true;
    }

    private void PetSurface_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
    {
        if (!_isMouseDown)
        {
            return;
        }

        PetSurface.ReleaseMouseCapture();
        _isMouseDown = false;
        if (!_isDragging)
        {
            SwitchToNextState();
        }

        e.Handled = true;
    }

    private void PetSurface_MouseRightButtonUp(object sender, MouseButtonEventArgs e)
    {
        PetSurface.ReleaseMouseCapture();
        _isMouseDown = false;
        _isDragging = false;

        var menu = new ContextMenu();
        var changeCharacter = new MenuItem
        {
            Header = "\u66f4\u6362\u89d2\u8272",
            IsEnabled = false
        };
        var exit = new MenuItem
        {
            Header = "\u9000\u51fa"
        };
        exit.Click += (_, _) => Application.Current.Shutdown();
        menu.Items.Add(changeCharacter);
        menu.Items.Add(new Separator());
        menu.Items.Add(exit);
        menu.IsOpen = true;
        e.Handled = true;
    }
}
