import 'package:dream11/login.dart';
import 'package:dream11/signup.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;
import 'package:file_picker/file_picker.dart'; 
import 'package:fl_chart/fl_chart.dart';


void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Optimal XI Predictor',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      // home: MyHomePage(),
      home: SignUp(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  String _responseText = 'Select a JSON file or press the button to get Optimal XI...';
  bool _isLoading = false;
  String _statusMessage = 'Select a JSON file or press the button to get Optimal XI...';
  Map<String, dynamic>? _matchData; // variable to hold the loaded match data
  String? _selectedFileName; 
  List<dynamic>? _optimalXi;
  double _totalPredictedPoints = 0.0;
  double _totalCredits = 0.0;
  String _roleCountsStr = "";
  String _teamSplitStr = "";
  Map<String, int> _roleCounts = {};
  Map<String, int> _teamSplit = {};
  final String _flaskEndpoint = 'http://127.0.0.1:5001/predict_team';

  Future<void> _pickJsonFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom, 
      allowedExtensions: ['json'], 
      withData: false, 
      withReadStream: true, 
    );

    if (result != null) {
      PlatformFile file = result.files.first;

      setState(() {
        _isLoading = true;
        _selectedFileName = file.name;
        _responseText = 'Reading ${file.name}...';
      });

      try {
        final stream = file.readStream;
        if (stream == null) {
           throw Exception("Could not read file stream.");
        }
        final fileContent = await stream.transform(utf8.decoder).join();

        final Map<String, dynamic> parsedJson = jsonDecode(fileContent);

        setState(() {
          _matchData = parsedJson;
          _responseText = 'Successfully loaded and parsed ${file.name}.\nPress "Get Optimal XI" to send data.';
        });
        print("Successfully loaded and parsed ${file.name}.");

      } catch (e) {
        setState(() {
          _responseText = 'Error loading or parsing file: ${e.toString()}';
          _matchData = null; 
          _selectedFileName = null;
        });
        print("Error loading or parsing file: $e");
      } finally {
        setState(() {
          _isLoading = false;
        });
      }

    } else {
      setState(() {
         _responseText = 'File picking canceled.';
      });
      print("File picking canceled.");
    }
  }

  Future<void> _sendMatchData() async {
    if (_matchData == null) {
      setState(() {
        _statusMessage = 'Match data not loaded yet. Please select a JSON file.';
        _optimalXi = null; 
        _roleCounts = {}; 
        _teamSplit = {}; 
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _statusMessage = 'Sending data to Flask...';
      _optimalXi = null; 
      _roleCounts = {}; 
      _teamSplit = {}; 
    });

    try {
      final requestBody = jsonEncode(_matchData);
      print('Attempting to send data to URL: $_flaskEndpoint');
      final response = await http.post(
        Uri.parse(_flaskEndpoint),
        headers: <String, String>{
          'Content-Type': 'application/json; charset=UTF-8',
        },
        body: jsonEncode(_matchData),
      );

      if (response.statusCode == 200) {
        final Map<String, dynamic> responseBody = jsonDecode(response.body);

        final List<dynamic>? optimalXi = responseBody['optimal_xi'];
        final double totalPredictedPoints =
            (responseBody['total_predicted_points'] as num?)?.toDouble() ?? 0.0;

        double totalCredits = 0.0;
        Map<String, int> roleCounts = {};
        Map<String, int> teamSplit = {};

        if (optimalXi != null && optimalXi.isNotEmpty) {
          for (var player in optimalXi) {
            if (player is Map<String, dynamic>) {
              final double credits = (player['credit'] as num?)?.toDouble() ?? 0.0;
              final String role = player['role'] ?? 'N/A';
              final String team = player['team'] ?? 'N/A';

              totalCredits += credits;
              roleCounts[role] = (roleCounts[role] ?? 0) + 1;
              teamSplit[team] = (teamSplit[team] ?? 0) + 1;
            }
          }

          // this is for formatting summary strings for display
          String roleCountsStr = "Roles: ";
          roleCounts.forEach((role, count) {
            roleCountsStr += "$role: $count  ";
          });

          String teamSplitStr = "Team Split: ";
          teamSplit.forEach((team, count) {
            teamSplitStr += "$team: $count  ";
          });

          // update state with all results
          setState(() {
            _optimalXi = optimalXi;
            _totalPredictedPoints = totalPredictedPoints;
            _totalCredits = totalCredits;
            _roleCountsStr = roleCountsStr.trim();
            _teamSplitStr = teamSplitStr.trim();
            _statusMessage = "Optimal XI Generated Successfully!"; 
            
            _roleCounts = roleCounts; 
            _teamSplit = teamSplit; 
          });
          print("Successfully received and parsed optimal XI from Flask.");

        } else {
          // handle case where response is 200 but optimal_xi is missing or empty
          setState(() {
            _statusMessage = "Could not parse optimal XI from response.";
            _optimalXi = null;
            _roleCounts = {}; 
            _teamSplit = {}; 
          });
        }
      } else {
        // handle HTTP error
        setState(() {
          _statusMessage =
              'Error (Status ${response.statusCode}):\n${response.body}';
          _optimalXi = null;
          _roleCounts = {}; 
          _teamSplit = {};
        });
        print(
            "Error response from Flask: Status ${response.statusCode}, Body: ${response.body}");
      }
    } catch (e) {
      // handle network or parsing errors
      setState(() {
        _statusMessage = 'An error occurred: ${e.toString()}';
        _optimalXi = null;
        _roleCounts = {}; 
        _teamSplit = {};
      });
      print("Network or other error: $e");
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Color _getRoleColor(String role) {
    switch (role) {
      case 'WK':
        return Colors.red.shade400;
      case 'BAT':
        return Colors.red.shade500;
      case 'AR':
        return Colors.red.shade600;
      case 'BOWL':
        return Colors.red.shade800;
      default:
        return Colors.red.shade900;
    }
  }

  Widget _buildRolePieChart() {
    if (_roleCounts.isEmpty) return SizedBox.shrink(); 

    final List<PieChartSectionData> sections = [];
    _roleCounts.forEach((role, count) {
      final section = PieChartSectionData(
        color: _getRoleColor(role),
        value: count.toDouble(),
        title: '$role\n($count)',
        radius: 60,
        titleStyle: TextStyle(
          fontSize: 12,
          fontWeight: FontWeight.bold,
          color: Colors.white,
          shadows: [Shadow(color: Colors.black, blurRadius: 2)],
        ),
      );
      sections.add(section);
    });

    return SizedBox(
      height: 180,
      child: Column(
        children: [
          Text("Role Split", style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
          SizedBox(height: 30),
          Expanded(
            child: PieChart(
              PieChartData(
                sections: sections,
                sectionsSpace: 2, 
                centerSpaceRadius: 30, 
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTeamPieChart() {
    if (_teamSplit.isEmpty) return SizedBox.shrink(); 
    final List<Color> teamColors = [Colors.red.shade900, Colors.red.shade500];
    final List<PieChartSectionData> sections = [];
    int i = 0;
    _teamSplit.forEach((team, count) {
      final section = PieChartSectionData(
        color: teamColors[i % teamColors.length], 
        value: count.toDouble(),
        title: '$team\n($count)',
        radius: 60,
        titleStyle: TextStyle(
          fontSize: 12,
          fontWeight: FontWeight.bold,
          color: Colors.white,
          shadows: [Shadow(color: Colors.black, blurRadius: 2)],
        ),
      );
      sections.add(section);
      i++;
    });

    return SizedBox(
      height: 180,
      child: Column(
        children: [
          Text("Team Split", style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
          SizedBox(height: 30),
          Expanded(
            child: PieChart(
              PieChartData(
                sections: sections,
                sectionsSpace: 2,
                centerSpaceRadius: 30,
              ),
            ),
          ),
        ],
      ),
    );
  }
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        title: Padding(
          padding: const EdgeInsets.only(top: 8.0),
          child: Text(
            "PredictMyXI",
            style: TextStyle(
              fontSize: 40,
              fontWeight: FontWeight.w700,
              color: Colors.red[800]
            ),
          ),
        ),
      ),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton(
                onPressed: _isLoading ? null : _pickJsonFile, 
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white, 
                  foregroundColor: Colors.grey[850],
                  side: BorderSide(color: Colors.red[800]!, width: 1), 
                  textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold), 
                ),
                child: _isLoading && _selectedFileName == null
                    ? const CircularProgressIndicator()
                    : const Text('Select Match JSON File'),
                ),
                SizedBox(width: 10),
                ElevatedButton(
                  onPressed: _isLoading || _matchData == null ? null : _sendMatchData,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red[800], 
                    foregroundColor: Colors.white,   
                  textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold), 
                ),
                child: _isLoading && _selectedFileName != null 
                ? const CircularProgressIndicator(
                color: Colors.white,
              )
              : const Text('Predict'),
                  ),
                ],
              ),
              
              SizedBox(height: 16),
              if (_selectedFileName != null && !_isLoading) 
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 8.0),
                  child: Text(
                    'Selected File: $_selectedFileName',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: Colors.red[800]
                    ),
                    ),
                ),
              SizedBox(height: 12),

              // 1. Status Message (replaces _responseText)
              // This will show errors, loading messages, or success messages
              Text(
                _statusMessage,
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  // Make errors red
                  color: _optimalXi == null && (_statusMessage.startsWith('Error') || _statusMessage.startsWith('An error')) 
                          ? Colors.red 
                          : Colors.black87,
                ),
              ),
              SizedBox(height: 16),

              // 2. Conditional Results Block
              // This whole block only appears if _optimalXi has data
              if (_optimalXi != null && _optimalXi!.isNotEmpty)
                Container(
                  width: double.infinity,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start, 
                    children: [
                      //The Data Table
                      Center(
                        child: DataTable(
                          columnSpacing: 20,
                          headingRowColor: MaterialStateColor.resolveWith((states) => Colors.grey[100]!),
                          headingTextStyle: TextStyle(fontWeight: FontWeight.bold, color: Colors.black),
                          columns: [
                            DataColumn(label: Text('Player',style: TextStyle(fontSize: 16,color: Colors.red[800]))),
                            DataColumn(label: Text('Team',style: TextStyle(fontSize: 16,color: Colors.red[800]))),
                            DataColumn(label: Text('Role',style: TextStyle(fontSize: 16,color: Colors.red[800]))),
                            DataColumn(label: Text('Credits',style: TextStyle(fontSize: 16,color: Colors.red[800]))),
                            DataColumn(label: Text('Pred. FP',style: TextStyle(fontSize: 16,color: Colors.red[800]))),
                          ],
                          rows: _optimalXi!.map((player) {
                            final playerMap = player as Map<String, dynamic>;
                            return DataRow(cells: [
                              DataCell(Text(playerMap['player_name']?.toString() ?? 'N/A')),
                              DataCell(Text(playerMap['team']?.toString() ?? 'N/A')),
                              DataCell(Text(playerMap['role']?.toString() ?? 'N/A')),
                              DataCell(Text((playerMap['credit'] as num?)?.toStringAsFixed(1) ?? '0.0')),
                              DataCell(Text((playerMap['predicted_fantasy_points'] as num?)?.toStringAsFixed(2) ?? '0.0')),
                            ]);
                          }).toList(), 
                        ),
                      ),
                      SizedBox(height: 24), 
                      Center(
                        child: Text(
                          'Total Predicted FP: ${_totalPredictedPoints.toStringAsFixed(2)}',
                          style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                        ),
                      ),
                      SizedBox(height: 4),
                      Center(
                        child: Text(
                          'Total Credits: ${_totalCredits.toStringAsFixed(1)}',
                          style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                        ),
                      ),
                      SizedBox(height: 24),
                      Center(
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.spaceAround,
                          children: [
                            Expanded(child: _buildRolePieChart()),
                            Expanded(child: _buildTeamPieChart()),
                          ],
                        ),
                      ),
                      SizedBox(height: 24),
                      
                      // Text(_teamSplitStr, style: TextStyle(fontSize: 14, color: Colors.grey[800])),
                      // SizedBox(height: 4),
                      // Text(_roleCountsStr, style: TextStyle(fontSize: 14, color: Colors.grey[800])),
                      // SizedBox(height: 16),
                    ],
                  ),
                )
            ],
          ),
        ),
      ),
    );
  }
}
