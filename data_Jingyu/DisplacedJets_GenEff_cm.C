#ifdef __CLING__
#pragma cling optimize(0)
#endif
void DisplacedJets_GenEff_cm()
{
//=========Macro generated from canvas: can/can
//=========  (Wed Dec 17 20:20:01 2025) by ROOT version 6.34.04
   TCanvas *can = new TCanvas("can", "can",0,62,800,800);
   gStyle->SetOptStat(0);
   can->SetHighLightColor(2);
   can->Range(-2.935517,-0.2763158,3.665351,1.697368);
   can->SetFillColor(0);
   can->SetBorderMode(0);
   can->SetBorderSize(2);
   can->SetLogx();
   can->SetTickx(1);
   can->SetTicky(1);
   can->SetLeftMargin(0.15);
   can->SetRightMargin(0.14);
   can->SetBottomMargin(0.14);
   can->SetFrameBorderMode(0);
   can->SetFrameBorderMode(0);
   
   Double_t eff_graph_fx3001[46] = { 0.0142, 0.01785, 0.0225, 0.02835, 0.0357, 0.04495, 0.0566, 0.07125, 0.0897, 0.113, 0.142, 0.1785, 0.225, 0.2835, 0.357, 0.4495, 0.566,
   0.7125, 0.897, 1.13, 1.42, 1.785, 2.25, 2.835, 3.57, 4.495, 5.66, 7.125, 8.97, 11.3, 14.2, 17.85, 22.5,
   28.35, 35.7, 44.95, 56.6, 71.25, 89.7, 113, 142, 178.5, 225, 283.5, 357, 449.5 };
   Double_t eff_graph_fy3001[46] = { 0.09643047, 0.08952145, 0.1009836, 0.11388, 0.1108719, 0.1117754, 0.1243624, 0.149535, 0.1671607, 0.1954218, 0.2277989, 0.2714901, 0.3177994, 0.3655153, 0.4089655, 0.4550722, 0.4914377,
   0.5298314, 0.5530303, 0.5727027, 0.5969724, 0.6224102, 0.63338, 0.6447852, 0.6424876, 0.646806, 0.6523198, 0.6596828, 0.6492092, 0.6359702, 0.6410608, 0.6447846, 0.6452564,
   0.6415889, 0.6540908, 0.6331849, 0.615738, 0.6039617, 0.5740138, 0.5440563, 0.4620376, 0.3210927, 0.2864611, 0.1906634, 0.02989537, 0.02436054 };
   Double_t eff_graph_felx3001[46] = { 0.0016, 0.00205, 0.0026, 0.00325, 0.0041, 0.00515, 0.0065, 0.00815, 0.0103, 0.013, 0.016, 0.0205, 0.026, 0.0325, 0.041, 0.0515, 0.065,
   0.0815, 0.103, 0.13, 0.16, 0.205, 0.26, 0.325, 0.41, 0.515, 0.65, 0.815, 1.03, 1.3, 1.6, 2.05, 2.6,
   3.25, 4.1, 5.15, 6.5, 8.15, 10.3, 13, 16, 20.5, 26, 32.5, 41, 51.5 };
   Double_t eff_graph_fely3001[46] = { 0.006856603, 0.005830028, 0.005485109, 0.005227766, 0.004628717, 0.004259709, 0.004059356, 0.003962438, 0.003856161, 0.003759281, 0.003752367, 0.003696017, 0.003641373, 0.003674838, 0.003602481, 0.003605075, 0.003604783,
   0.003589527, 0.003660897, 0.003667892, 0.003699297, 0.003603905, 0.003571609, 0.003501973, 0.003463142, 0.003411502, 0.003352845, 0.003363286, 0.003381619, 0.003445205, 0.00345166, 0.003383808, 0.00333078,
   0.003256335, 0.003201795, 0.00322809, 0.003213456, 0.003275056, 0.003351926, 0.003436064, 0.003415618, 0.003138726, 0.003159202, 0.003570243, 0.001987607, 0.001794983 };
   Double_t eff_graph_fehx3001[46] = { 0.0016, 0.00205, 0.0026, 0.00325, 0.0041, 0.00515, 0.0065, 0.00815, 0.0103, 0.013, 0.016, 0.0205, 0.026, 0.0325, 0.041, 0.0515, 0.065,
   0.0815, 0.103, 0.13, 0.16, 0.205, 0.26, 0.325, 0.41, 0.515, 0.65, 0.815, 1.03, 1.3, 1.6, 2.05, 2.6,
   3.25, 4.1, 5.15, 6.5, 8.15, 10.3, 13, 16, 20.5, 26, 32.5, 41, 51.5 };
   Double_t eff_graph_fehy3001[46] = { 0.007303303, 0.006180678, 0.005754574, 0.005439991, 0.004800159, 0.00440336, 0.004174141, 0.004049534, 0.003927601, 0.003814182, 0.003795989, 0.00372756, 0.00366361, 0.003690444, 0.003612212, 0.003609757, 0.003605667,
   0.003586466, 0.003655197, 0.003659974, 0.003688381, 0.003590526, 0.003557124, 0.003486651, 0.003448435, 0.003396717, 0.003337921, 0.003347388, 0.003366809, 0.00343142, 0.00343722, 0.003369497, 0.003316858,
   0.003243419, 0.003187989, 0.003216265, 0.003203452, 0.003265821, 0.003345184, 0.003431903, 0.003419161, 0.003154865, 0.003180033, 0.003621493, 0.002119738, 0.001928579 };
   TGraphAsymmErrors *grae = new TGraphAsymmErrors(46,eff_graph_fx3001,eff_graph_fy3001,eff_graph_felx3001,eff_graph_fehx3001,eff_graph_fely3001,eff_graph_fehy3001);
   grae->SetName("eff_graph");
   grae->SetTitle("");
   grae->SetFillColor(19);

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
   ci = TColor::GetColor("#0000ff");
   grae->SetLineColor(ci);
   grae->SetLineWidth(2);

   ci = TColor::GetColor("#0000ff");
   grae->SetMarkerColor(ci);
   grae->SetMarkerStyle(20);
   grae->SetMarkerSize(1.5);
   
   TH1F *Graph_eff_graph3001 = new TH1F("Graph_eff_graph3001","",100,0.01134,551.0987);
   Graph_eff_graph3001->SetMinimum(0);
   Graph_eff_graph3001->SetMaximum(1.5);
   Graph_eff_graph3001->SetDirectory(nullptr);
   Graph_eff_graph3001->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_eff_graph3001->SetLineColor(ci);
   Graph_eff_graph3001->GetXaxis()->SetTitle("Gen.-level parton production vertex L_{xy} [cm]");
   Graph_eff_graph3001->GetXaxis()->SetLabelFont(42);
   Graph_eff_graph3001->GetXaxis()->SetTitleSize(0.042);
   Graph_eff_graph3001->GetXaxis()->SetTitleOffset(1.3);
   Graph_eff_graph3001->GetXaxis()->SetTitleFont(42);
   Graph_eff_graph3001->GetYaxis()->SetTitle("HLT per-parton tagging efficiency");
   Graph_eff_graph3001->GetYaxis()->SetLabelFont(42);
   Graph_eff_graph3001->GetYaxis()->SetLabelSize(0);
   Graph_eff_graph3001->GetYaxis()->SetTitleSize(0.045);
   Graph_eff_graph3001->GetYaxis()->SetTickLength(0);
   Graph_eff_graph3001->GetYaxis()->SetTitleOffset(1.4);
   Graph_eff_graph3001->GetYaxis()->SetTitleFont(42);
   Graph_eff_graph3001->GetZaxis()->SetLabelFont(42);
   Graph_eff_graph3001->GetZaxis()->SetTitleOffset(1);
   Graph_eff_graph3001->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_eff_graph3001);
   
   grae->Draw("ape");
   TGaxis *gaxis = new TGaxis(0.01134,0,0.01134,1,0,1,505,"S");
   gaxis->SetLabelOffset(0.005);
   gaxis->SetLabelSize(0.035);
   gaxis->SetTickSize(0.03);
   gaxis->SetGridLength(0);
   gaxis->SetTitleOffset(1);
   gaxis->SetTitleSize(0.04);
   gaxis->SetTitleColor(1);
   gaxis->SetTitleFont(62);
   gaxis->SetLabelFont(42);
   gaxis->Draw();
   gaxis = new TGaxis(551.0987,0,551.0987,1,0,1,505,"+S");
   gaxis->SetLabelOffset(0.005);
   gaxis->SetLabelSize(0);
   gaxis->SetTickSize(0.03);
   gaxis->SetGridLength(0);
   gaxis->SetTitleOffset(1);
   gaxis->SetTitleSize(0.04);
   gaxis->SetTitleColor(1);
   gaxis->SetTitleFont(62);
   gaxis->Draw();
   
   Double_t eff_graph_fx3002[46] = { 0.0142, 0.01785, 0.0225, 0.02835, 0.0357, 0.04495, 0.0566, 0.07125, 0.0897, 0.113, 0.142, 0.1785, 0.225, 0.2835, 0.357, 0.4495, 0.566,
   0.7125, 0.897, 1.13, 1.42, 1.785, 2.25, 2.835, 3.57, 4.495, 5.66, 7.125, 8.97, 11.3, 14.2, 17.85, 22.5,
   28.35, 35.7, 44.95, 56.6, 71.25, 89.7, 113, 142, 178.5, 225, 283.5, 357, 449.5 };
   Double_t eff_graph_fy3002[46] = { 0.3335659, 0.3167496, 0.3445161, 0.3560924, 0.3831858, 0.3854191, 0.4060403, 0.4024554, 0.4320212, 0.4361307, 0.456565, 0.4686887, 0.4794038, 0.4852764, 0.4958127, 0.5044326, 0.5029011,
   0.5098321, 0.5077086, 0.5038346, 0.5072263, 0.5082625, 0.5108671, 0.5176409, 0.5221776, 0.5194143, 0.5211287, 0.514234, 0.4963638, 0.5030214, 0.4966755, 0.4947747, 0.4908437,
   0.4856711, 0.4883056, 0.4823888, 0.4681236, 0.4726362, 0.4534413, 0.4159661, 0.3672275, 0.2662572, 0.2128638, 0.1100114, 0.02611049, 0.02302952 };
   Double_t eff_graph_felx3002[46] = { 0.0016, 0.00205, 0.0026, 0.00325, 0.0041, 0.00515, 0.0065, 0.00815, 0.0103, 0.013, 0.016, 0.0205, 0.026, 0.0325, 0.041, 0.0515, 0.065,
   0.0815, 0.103, 0.13, 0.16, 0.205, 0.26, 0.325, 0.41, 0.515, 0.65, 0.815, 1.03, 1.3, 1.6, 2.05, 2.6,
   3.25, 4.1, 5.15, 6.5, 8.15, 10.3, 13, 16, 20.5, 26, 32.5, 41, 51.5 };
   Double_t eff_graph_fely3002[46] = { 0.01268042, 0.01110747, 0.01000008, 0.009082297, 0.008461119, 0.007590253, 0.006978907, 0.006426561, 0.005912271, 0.005477757, 0.005146497, 0.004754749, 0.0045105, 0.004367995, 0.004176314, 0.004163086, 0.004140644,
   0.004172731, 0.004230266, 0.004309711, 0.00442231, 0.004432229, 0.00440412, 0.004372778, 0.004307935, 0.004269862, 0.004200765, 0.004205412, 0.004236166, 0.004226307, 0.004262995, 0.004193462, 0.00411203,
   0.004083784, 0.004060645, 0.004011216, 0.00404245, 0.004058003, 0.004085631, 0.004171981, 0.004149573, 0.003776079, 0.003661854, 0.003349226, 0.002014036, 0.001911538 };
   Double_t eff_graph_fehx3002[46] = { 0.0016, 0.00205, 0.0026, 0.00325, 0.0041, 0.00515, 0.0065, 0.00815, 0.0103, 0.013, 0.016, 0.0205, 0.026, 0.0325, 0.041, 0.0515, 0.065,
   0.0815, 0.103, 0.13, 0.16, 0.205, 0.26, 0.325, 0.41, 0.515, 0.65, 0.815, 1.03, 1.3, 1.6, 2.05, 2.6,
   3.25, 4.1, 5.15, 6.5, 8.15, 10.3, 13, 16, 20.5, 26, 32.5, 41, 51.5 };
   Double_t eff_graph_fehy3002[46] = { 0.01291911, 0.01131512, 0.01013671, 0.00918502, 0.00853124, 0.007645527, 0.007016527, 0.006459802, 0.005931473, 0.005493217, 0.005155693, 0.004760388, 0.004513831, 0.004370226, 0.004176894, 0.004162476, 0.00414025,
   0.004171373, 0.004229171, 0.004309146, 0.00442119, 0.004430942, 0.004402448, 0.004370101, 0.004304666, 0.004267052, 0.004197804, 0.004203414, 0.004236684, 0.004225879, 0.004263475, 0.004194191, 0.00411326,
   0.004085683, 0.004062177, 0.004013469, 0.004046605, 0.004061593, 0.004091862, 0.004183956, 0.00416917, 0.00381018, 0.003707892, 0.003439468, 0.002170896, 0.002072811 };
   grae = new TGraphAsymmErrors(46,eff_graph_fx3002,eff_graph_fy3002,eff_graph_felx3002,eff_graph_fehx3002,eff_graph_fely3002,eff_graph_fehy3002);
   grae->SetName("eff_graph");
   grae->SetTitle("eff_dxy");
   grae->SetFillColor(19);

   ci = TColor::GetColor("#00cc00");
   grae->SetLineColor(ci);
   grae->SetLineWidth(2);

   ci = TColor::GetColor("#00cc00");
   grae->SetMarkerColor(ci);
   grae->SetMarkerStyle(21);
   grae->SetMarkerSize(1.5);
   
   TH1F *Graph_eff_graph3002 = new TH1F("Graph_eff_graph3002","eff_dxy",100,0.01134,551.0987);
   Graph_eff_graph3002->SetMinimum(0.01900618);
   Graph_eff_graph3002->SetMaximum(0.5770187);
   Graph_eff_graph3002->SetDirectory(nullptr);
   Graph_eff_graph3002->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_eff_graph3002->SetLineColor(ci);
   Graph_eff_graph3002->GetXaxis()->SetLabelFont(42);
   Graph_eff_graph3002->GetXaxis()->SetTitleOffset(1);
   Graph_eff_graph3002->GetXaxis()->SetTitleFont(42);
   Graph_eff_graph3002->GetYaxis()->SetLabelFont(42);
   Graph_eff_graph3002->GetYaxis()->SetTitleFont(42);
   Graph_eff_graph3002->GetZaxis()->SetLabelFont(42);
   Graph_eff_graph3002->GetZaxis()->SetTitleOffset(1);
   Graph_eff_graph3002->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_eff_graph3002);
   
   grae->Draw("pe ");
   
   Double_t eff_graph_fx3003[46] = { 0.0142, 0.01785, 0.0225, 0.02835, 0.0357, 0.04495, 0.0566, 0.07125, 0.0897, 0.113, 0.142, 0.1785, 0.225, 0.2835, 0.357, 0.4495, 0.566,
   0.7125, 0.897, 1.13, 1.42, 1.785, 2.25, 2.835, 3.57, 4.495, 5.66, 7.125, 8.97, 11.3, 14.2, 17.85, 22.5,
   28.35, 35.7, 44.95, 56.6, 71.25, 89.7, 113, 142, 178.5, 225, 283.5, 357, 449.5 };
   Double_t eff_graph_fy3003[46] = { 0.04819277, 0.04685019, 0.05319823, 0.06453261, 0.07735192, 0.09305882, 0.1147479, 0.155739, 0.1797723, 0.2301528, 0.2758113, 0.3210585, 0.3686049, 0.4234747, 0.4787917, 0.5181586, 0.560483,
   0.5906295, 0.6304419, 0.6474918, 0.6762645, 0.6946852, 0.709051, 0.7175736, 0.723119, 0.7215544, 0.7244613, 0.7130345, 0.7146556, 0.7136494, 0.7050246, 0.7073272, 0.7038751,
   0.6981207, 0.6934805, 0.6787523, 0.651011, 0.6273814, 0.5947416, 0.5604578, 0.4630777, 0.3072114, 0.2832065, 0.184769, 0.03834449, 0.03494386 };
   Double_t eff_graph_felx3003[46] = { 0.0016, 0.00205, 0.0026, 0.00325, 0.0041, 0.00515, 0.0065, 0.00815, 0.0103, 0.013, 0.016, 0.0205, 0.026, 0.0325, 0.041, 0.0515, 0.065,
   0.0815, 0.103, 0.13, 0.16, 0.205, 0.26, 0.325, 0.41, 0.515, 0.65, 0.815, 1.03, 1.3, 1.6, 2.05, 2.6,
   3.25, 4.1, 5.15, 6.5, 8.15, 10.3, 13, 16, 20.5, 26, 32.5, 41, 51.5 };
   Double_t eff_graph_fely3003[46] = { 0.003929018, 0.003446694, 0.003269544, 0.003215385, 0.003163482, 0.003161183, 0.003101231, 0.003258469, 0.003135719, 0.003209182, 0.003198764, 0.003127751, 0.00307184, 0.003029438, 0.002975664, 0.002950858, 0.00295597,
   0.002961768, 0.002933873, 0.002967472, 0.002988055, 0.002911549, 0.002869626, 0.002813339, 0.002765438, 0.002744307, 0.002714849, 0.002776243, 0.002772934, 0.00280432, 0.002863494, 0.002828093, 0.002803534,
   0.002787726, 0.002753884, 0.002792982, 0.002877367, 0.002946689, 0.00303986, 0.003140016, 0.003160165, 0.002868937, 0.002949333, 0.003339182, 0.002122057, 0.002065597 };
   Double_t eff_graph_fehx3003[46] = { 0.0016, 0.00205, 0.0026, 0.00325, 0.0041, 0.00515, 0.0065, 0.00815, 0.0103, 0.013, 0.016, 0.0205, 0.026, 0.0325, 0.041, 0.0515, 0.065,
   0.0815, 0.103, 0.13, 0.16, 0.205, 0.26, 0.325, 0.41, 0.515, 0.65, 0.815, 1.03, 1.3, 1.6, 2.05, 2.6,
   3.25, 4.1, 5.15, 6.5, 8.15, 10.3, 13, 16, 20.5, 26, 32.5, 41, 51.5 };
   Double_t eff_graph_fehy3003[46] = { 0.004244303, 0.00369579, 0.003464276, 0.003367606, 0.003283888, 0.003258716, 0.0031748, 0.003314335, 0.003178558, 0.003240577, 0.003221716, 0.003143781, 0.003082462, 0.003035167, 0.002977161, 0.0029496, 0.002951711,
   0.002955246, 0.002924326, 0.002956207, 0.002973843, 0.002896173, 0.002853146, 0.002796561, 0.002748611, 0.002727907, 0.002698483, 0.002760392, 0.002756947, 0.002788081, 0.002847525, 0.002812269, 0.002788343,
   0.002773285, 0.002740234, 0.002780331, 0.002866471, 0.002937315, 0.003032657, 0.003135214, 0.003163116, 0.002883826, 0.002967895, 0.003385991, 0.002237674, 0.002186515 };
   grae = new TGraphAsymmErrors(46,eff_graph_fx3003,eff_graph_fy3003,eff_graph_felx3003,eff_graph_fehx3003,eff_graph_fely3003,eff_graph_fehy3003);
   grae->SetName("eff_graph");
   grae->SetTitle("eff_dxy");
   grae->SetFillColor(19);
   grae->SetLineColor(6);
   grae->SetLineWidth(2);
   grae->SetMarkerColor(6);
   grae->SetMarkerStyle(22);
   grae->SetMarkerSize(1.5);
   
   TH1F *Graph_eff_graph3003 = new TH1F("Graph_eff_graph3003","eff_dxy",100,0.01134,551.0987);
   Graph_eff_graph3003->SetMinimum(0.02959044);
   Graph_eff_graph3003->SetMaximum(0.796588);
   Graph_eff_graph3003->SetDirectory(nullptr);
   Graph_eff_graph3003->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_eff_graph3003->SetLineColor(ci);
   Graph_eff_graph3003->GetXaxis()->SetLabelFont(42);
   Graph_eff_graph3003->GetXaxis()->SetTitleOffset(1);
   Graph_eff_graph3003->GetXaxis()->SetTitleFont(42);
   Graph_eff_graph3003->GetYaxis()->SetLabelFont(42);
   Graph_eff_graph3003->GetYaxis()->SetTitleFont(42);
   Graph_eff_graph3003->GetZaxis()->SetLabelFont(42);
   Graph_eff_graph3003->GetZaxis()->SetTitleOffset(1);
   Graph_eff_graph3003->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_eff_graph3003);
   
   grae->Draw("pe ");
   TLatex *   tex = new TLatex(0.18,0.82,"CMS #bf{#scale[0.75]{#it{Simulation}}}");
   tex->SetNDC();
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.2,0.72,"#splitline{H #rightarrow SS, S #rightarrow b#bar{b}/d#bar{d}/#tau^{+}#tau^{-}}{m_{H} = 125 GeV, m_{S} = 40 GeV}");
   tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetTextSize(0.03);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.2,0.6,"#splitline{b/d/#tau}{Gen.-level p_{T} > 40 GeV, |#eta| < 2.0}");
   tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetTextSize(0.03);
   tex->SetLineWidth(2);
   tex->Draw();
   
   TLegend *leg = new TLegend(0.6,0.55,0.83,0.78,NULL,"brNDC");
   leg->SetBorderSize(1);
   leg->SetLineColor(0);
   leg->SetLineStyle(1);
   leg->SetLineWidth(1);
   leg->SetFillColor(0);
   leg->SetFillStyle(1001);
   TLegendEntry *entry=leg->AddEntry("eff_graph","b quark","ple");

   ci = TColor::GetColor("#0000ff");
   entry->SetLineColor(ci);
   entry->SetLineStyle(1);
   entry->SetLineWidth(2);

   ci = TColor::GetColor("#0000ff");
   entry->SetMarkerColor(ci);
   entry->SetMarkerStyle(20);
   entry->SetMarkerSize(1.5);
   entry->SetTextFont(42);
   entry=leg->AddEntry("eff_graph","d quark","ple");
   entry->SetLineColor(6);
   entry->SetLineStyle(1);
   entry->SetLineWidth(2);
   entry->SetMarkerColor(6);
   entry->SetMarkerStyle(22);
   entry->SetMarkerSize(1.5);
   entry->SetTextFont(42);
   entry=leg->AddEntry("eff_graph","#tau lepton","ple");

   ci = TColor::GetColor("#00cc00");
   entry->SetLineColor(ci);
   entry->SetLineStyle(1);
   entry->SetLineWidth(2);

   ci = TColor::GetColor("#00cc00");
   entry->SetMarkerColor(ci);
   entry->SetMarkerStyle(21);
   entry->SetMarkerSize(1.5);
   entry->SetTextFont(42);
   leg->Draw();
      tex = new TLatex(0.65,0.92,"(13.6 TeV)");
   tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetLineWidth(2);
   tex->Draw();
   can->Modified();
   can->SetSelected(can);
}
