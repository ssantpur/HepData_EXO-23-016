#ifdef __CLING__
#pragma cling optimize(0)
#endif
void Figure39_lower_canvas()
{
//=========Macro generated from canvas: c/
//=========  (Thu Nov 20 14:00:47 2025) by ROOT version 6.30/07
   TCanvas *c = new TCanvas("c", "",1,1,1280,1000);
   c->SetHighLightColor(2);
   c->Range(-15,-0.1875,135,1.6875);
   c->SetFillColor(0);
   c->SetBorderMode(0);
   c->SetBorderSize(2);
   c->SetFrameBorderMode(0);
   c->SetFrameBorderMode(0);
   
   Double_t divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF1_fx3005[30] = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 36,
   40, 44, 48, 52.5, 57.5, 62.5, 67.5, 75, 85, 95, 110, 135, 175 };
   Double_t divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF1_fy3005[30] = { 0.894165, 0.8864307, 0.8887151, 0.8812691, 0.8767193, 0.8822009, 0.867173, 0.8551907, 0.8464146, 0.8338801, 0.836463, 0.8339265, 0.8202643, 0.8175944, 0.8084869, 0.8109535, 0.7840031,
   0.7723276, 0.7518046, 0.7381841, 0.7176829, 0.7163662, 0.6574307, 0.6594488, 0.6472065, 0.5889423, 0.5506329, 0.4604371, 0.315829, 0.1538462 };
   Double_t divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF1_felx3005[30] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
   2, 2, 2, 2.5, 2.5, 2.5, 2.5, 5, 5, 5, 10, 15, 25 };
   Double_t divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF1_fely3005[30] = { 0.001395258, 0.002268519, 0.002780796, 0.003298274, 0.003822556, 0.004109598, 0.004781229, 0.005414432, 0.006044547, 0.006678056, 0.007121793, 0.007732621, 0.008423118, 0.00901899, 0.009752805, 0.007381138, 0.008393497,
   0.009379439, 0.01059757, 0.01142348, 0.01155324, 0.01236774, 0.01430155, 0.0155143, 0.01237715, 0.01439588, 0.01672929, 0.01402342, 0.01296027, 0.01049057 };
   Double_t divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF1_fehx3005[30] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
   2, 2, 2, 2.5, 2.5, 2.5, 2.5, 5, 5, 5, 10, 15, 25 };
   Double_t divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF1_fehy3005[30] = { 0.001379341, 0.002230103, 0.002722095, 0.003221991, 0.00372496, 0.003991197, 0.004642475, 0.005254574, 0.005859869, 0.006474317, 0.006886354, 0.007461549, 0.008133309, 0.008694271, 0.009398092, 0.007171237, 0.008168829,
   0.009121007, 0.01031041, 0.01111894, 0.01128059, 0.01205901, 0.01402925, 0.01519023, 0.01218756, 0.01424931, 0.01661905, 0.01408466, 0.0132446, 0.01108573 };
   TGraphAsymmErrors *grae = new TGraphAsymmErrors(30,divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF1_fx3005,divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF1_fy3005,divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF1_felx3005,divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF1_fehx3005,divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF1_fely3005,divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF1_fehy3005);
   grae->SetName("divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF1");
   grae->SetTitle("");

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
   ci = TColor::GetColor("#3f90da");
   grae->SetLineColor(ci);

   ci = TColor::GetColor("#3f90da");
   grae->SetMarkerColor(ci);
   grae->SetMarkerStyle(21);
   //grae->SetMarkerSize(2.5);
   grae->SetMarkerSize(3);
   
   TH1F *Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005 = new TH1F("Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005","",100,0,120);
   Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005->SetMinimum(0);
   Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005->SetMaximum(1.5);
   Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005->SetDirectory(nullptr);
   Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005->SetLineColor(ci);
   Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005->GetXaxis()->SetTitle("Gen.-level muon track d_{0} [cm]");
   Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005->GetXaxis()->SetNdivisions(509);
   Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005->GetXaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005->GetXaxis()->SetTitleSize(0.045);
   Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005->GetXaxis()->SetTitleOffset(1);
   Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005->GetXaxis()->SetTitleFont(42);
   Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005->GetYaxis()->SetTitle("L1T efficiency");
   Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005->GetYaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005->GetYaxis()->SetTitleSize(0.045);
   Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005->GetYaxis()->SetTitleOffset(1);
   Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005->GetYaxis()->SetTitleFont(42);
   Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005->GetZaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005->GetZaxis()->SetTitleOffset(1);
   Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF13005);
   
   grae->Draw("ap");
   
   Double_t divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF1_fx3006[30] = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 36,
   40, 44, 48, 52.5, 57.5, 62.5, 67.5, 75, 85, 95, 110, 135, 175 };
   Double_t divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF1_fy3006[30] = { 0.9465614, 0.9353982, 0.9313222, 0.9159523, 0.9028273, 0.8995288, 0.8947758, 0.8648887, 0.8435463, 0.8168804, 0.7872994, 0.7145117, 0.6449339, 0.556163, 0.453378, 0.2992412, 0.137558,
   0.04627487, 0.01110494, 0.002487562, 0.001829268, 0.0006934813, 0.002518892, 0.000984252, 0.0006277464, 0.002403846, 0.003164557, 0, 0.006751688, 0.03639371 };
   Double_t divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF1_felx3006[30] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
   2, 2, 2, 2.5, 2.5, 2.5, 2.5, 5, 5, 5, 10, 15, 25 };
   Double_t divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF1_fely3006[30] = { 0.001026009, 0.001769972, 0.002252728, 0.002845195, 0.003458595, 0.003844372, 0.004343762, 0.005267272, 0.0060879, 0.006924408, 0.007831331, 0.009263781, 0.01032715, 0.01135059, 0.01201233, 0.008413851, 0.006819345,
   0.004531945, 0.002453942, 0.00119014, 0.0009953907, 0.0005736869, 0.001370564, 0.0008142332, 0.0005193067, 0.001307979, 0.00172178, 0, 0.002204827, 0.005393055 };
   Double_t divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF1_fehx3006[30] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
   2, 2, 2, 2.5, 2.5, 2.5, 2.5, 5, 5, 5, 10, 15, 25 };
   Double_t divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF1_fehy3006[30] = { 0.001007929, 0.00172655, 0.002187378, 0.002761764, 0.003354042, 0.003720436, 0.004194266, 0.005102949, 0.005904781, 0.006731208, 0.007630767, 0.009090446, 0.01019669, 0.01129353, 0.01206561, 0.008548922, 0.007107369,
   0.004973276, 0.003057815, 0.001962506, 0.001776172, 0.001592855, 0.002444112, 0.002259641, 0.001442025, 0.002332747, 0.003068651, 0.001386394, 0.003068255, 0.006218562 };
   grae = new TGraphAsymmErrors(30,divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF1_fx3006,divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF1_fy3006,divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF1_felx3006,divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF1_fehx3006,divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF1_fely3006,divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF1_fehy3006);
   grae->SetName("divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF1");
   grae->SetTitle("");

   ci = TColor::GetColor("#3f90da");
   grae->SetLineColor(ci);

   ci = TColor::GetColor("#3f90da");
   grae->SetMarkerColor(ci);
   grae->SetMarkerStyle(25);
   //grae->SetMarkerSize(2.5);
   grae->SetMarkerSize(3);
   
   TH1F *Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006 = new TH1F("Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006","",100,0,220);
   Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006->SetMinimum(0);
   Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006->SetMaximum(1.5);
   Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006->SetDirectory(nullptr);
   Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006->SetLineColor(ci);
   Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006->GetXaxis()->SetTitle("d_{0} (gen #mu) [cm]");
   Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006->GetXaxis()->SetRange(1,55);
   Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006->GetXaxis()->SetNdivisions(509);
   Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006->GetXaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006->GetXaxis()->SetTitleSize(1);
   Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006->GetXaxis()->SetTitleOffset(1);
   Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006->GetXaxis()->SetTitleFont(42);
   Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006->GetYaxis()->SetTitle("L1T efficiency");
   Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006->GetYaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006->GetYaxis()->SetTitleSize(1);
   Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006->GetYaxis()->SetTitleOffset(1.35);
   Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006->GetYaxis()->SetTitleFont(42);
   Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006->GetZaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006->GetZaxis()->SetTitleOffset(1);
   Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF13006);
   
   grae->Draw("s p");
   
   Double_t divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF2_fx3007[30] = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 36,
   40, 44, 48, 52.5, 57.5, 62.5, 67.5, 75, 85, 95, 110, 135, 175 };
   Double_t divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF2_fy3007[30] = { 0.6902156, 0.691689, 0.6820038, 0.7017437, 0.7048828, 0.6879718, 0.6647539, 0.6502339, 0.603871, 0.5722952, 0.5322953, 0.5234633, 0.4996215, 0.4578527, 0.4595701, 0.458731, 0.4156891,
   0.4137645, 0.4127144, 0.3677885, 0.3103058, 0.3065903, 0.2449612, 0.2330677, 0.2261905, 0.2111111, 0.1288344, 0.0983871, 0.01972686, 0.001876173 };
   Double_t divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF2_felx3007[30] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
   2, 2, 2, 2.5, 2.5, 2.5, 2.5, 5, 5, 5, 10, 15, 25 };
   Double_t divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF2_fely3007[30] = { 0.002419289, 0.003927655, 0.004967025, 0.00573991, 0.00651185, 0.007522743, 0.008404628, 0.009279934, 0.01040175, 0.01141145, 0.01228603, 0.01318211, 0.01412954, 0.01523909, 0.01640447, 0.01206095, 0.01364242,
   0.01451698, 0.01604457, 0.01714358, 0.01590365, 0.01786809, 0.01728456, 0.01929087, 0.01468414, 0.01656226, 0.01535814, 0.01207772, 0.005372084, 0.001552109 };
   Double_t divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF2_fehx3007[30] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
   2, 2, 2, 2.5, 2.5, 2.5, 2.5, 5, 5, 5, 10, 15, 25 };
   Double_t divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF2_fehy3007[30] = { 0.002408977, 0.003900362, 0.00492641, 0.005677911, 0.006430601, 0.007426538, 0.008303295, 0.009169687, 0.01031053, 0.01133672, 0.0122479, 0.01315031, 0.01413013, 0.01531608, 0.01648984, 0.01210838, 0.01376939,
   0.0146641, 0.0162263, 0.01747258, 0.01634862, 0.01844452, 0.0181108, 0.02040912, 0.01536246, 0.01752321, 0.01697669, 0.01345932, 0.007032439, 0.00430097 };
   grae = new TGraphAsymmErrors(30,divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF2_fx3007,divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF2_fy3007,divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF2_felx3007,divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF2_fehx3007,divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF2_fely3007,divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF2_fehy3007);
   grae->SetName("divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF2");
   grae->SetTitle("");

   ci = TColor::GetColor("#e42536");
   grae->SetLineColor(ci);

   ci = TColor::GetColor("#e42536");
   grae->SetMarkerColor(ci);
   grae->SetMarkerStyle(22);
   //grae->SetMarkerSize(2.5);
   grae->SetMarkerSize(3);
   
   TH1F *Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007 = new TH1F("Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007","",100,0,120);
   Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007->SetMinimum(0);
   Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007->SetMaximum(1.5);
   Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007->SetDirectory(nullptr);
   Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007->SetLineColor(ci);
   Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007->GetXaxis()->SetTitle("Gen.-level muon track d_{0} [cm]");
   Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007->GetXaxis()->SetNdivisions(509);
   Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007->GetXaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007->GetXaxis()->SetTitleSize(0.045);
   Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007->GetXaxis()->SetTitleOffset(1);
   Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007->GetXaxis()->SetTitleFont(42);
   Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007->GetYaxis()->SetTitle("L1T efficiency");
   Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007->GetYaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007->GetYaxis()->SetTitleSize(0.045);
   Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007->GetYaxis()->SetTitleOffset(1);
   Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007->GetYaxis()->SetTitleFont(42);
   Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007->GetZaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007->GetZaxis()->SetTitleOffset(1);
   Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF23007);
   
   grae->Draw("s p");
   
   Double_t divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF2_fx3008[30] = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 36,
   40, 44, 48, 52.5, 57.5, 62.5, 67.5, 75, 85, 95, 110, 135, 175 };
   Double_t divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF2_fy3008[30] = { 0.8800809, 0.8576972, 0.8354849, 0.8235027, 0.8083984, 0.7881228, 0.7478103, 0.70457, 0.6339785, 0.5687563, 0.4803922, 0.4203569, 0.339137, 0.2679681, 0.1873081, 0.157215, 0.0542522,
   0.01658375, 0.009081736, 0.008413462, 0, 0.00286533, 0.001550388, 0, 0.001190476, 0, 0.00204499, 0, 0, 0 };
   Double_t divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF2_felx3008[30] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
   2, 2, 2, 2.5, 2.5, 2.5, 2.5, 5, 5, 5, 10, 15, 25 };
   Double_t divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF2_fely3008[30] = { 0.001710426, 0.002995236, 0.003986965, 0.004820011, 0.005658801, 0.006681379, 0.007773496, 0.008908196, 0.01026209, 0.01142095, 0.01227122, 0.01296238, 0.01327383, 0.01341881, 0.01265689, 0.008706207, 0.006156847,
   0.003658506, 0.00296418, 0.003097665, 0, 0.001850525, 0.001282588, 0, 0.0009848381, 0, 0.001691772, 0, 0, 0 };
   Double_t divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF2_fehx3008[30] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
   2, 2, 2, 2.5, 2.5, 2.5, 2.5, 5, 5, 5, 10, 15, 25 };
   Double_t divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF2_fehy3008[30] = { 0.001689773, 0.002944159, 0.003911887, 0.00472033, 0.005536208, 0.006533583, 0.007620839, 0.008757942, 0.01014439, 0.01134988, 0.01229437, 0.01307036, 0.01352434, 0.0138442, 0.01332283, 0.009103543, 0.006849251,
   0.004550289, 0.004120083, 0.00450225, 0.00208279, 0.003766622, 0.003556047, 0.003660657, 0.002732159, 0.002917991, 0.004686662, 0.002964985, 0.002789761, 0.003448116 };
   grae = new TGraphAsymmErrors(30,divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF2_fx3008,divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF2_fy3008,divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF2_felx3008,divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF2_fehx3008,divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF2_fely3008,divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF2_fehy3008);
   grae->SetName("divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF2");
   grae->SetTitle("");

   ci = TColor::GetColor("#e42536");
   grae->SetLineColor(ci);

   ci = TColor::GetColor("#e42536");
   grae->SetMarkerColor(ci);
   grae->SetMarkerStyle(26);
   //grae->SetMarkerSize(2.5);
   grae->SetMarkerSize(3);
   
   TH1F *Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008 = new TH1F("Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008","",100,0,220);
   Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008->SetMinimum(0);
   Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008->SetMaximum(1.5);
   Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008->SetDirectory(nullptr);
   Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008->SetLineColor(ci);
   Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008->GetXaxis()->SetTitle("d_{0} (gen #mu) [cm]");
   Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008->GetXaxis()->SetRange(1,55);
   Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008->GetXaxis()->SetNdivisions(509);
   Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008->GetXaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008->GetXaxis()->SetTitleSize(1);
   Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008->GetXaxis()->SetTitleOffset(1);
   Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008->GetXaxis()->SetTitleFont(42);
   Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008->GetYaxis()->SetTitle("L1T efficiency");
   Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008->GetYaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008->GetYaxis()->SetTitleSize(1);
   Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008->GetYaxis()->SetTitleOffset(1.35);
   Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008->GetYaxis()->SetTitleFont(42);
   Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008->GetZaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008->GetZaxis()->SetTitleOffset(1);
   Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF23008);
   
   grae->Draw("s p");
   TGaxis *gaxis = new TGaxis(0,1.5,120,1.5,0,120,510,"-U");
   gaxis->SetLabelOffset(0.005);
   gaxis->SetLabelSize(0.04);
   gaxis->SetTickSize(0.03);
   gaxis->SetGridLength(0);
   gaxis->SetTitleOffset(1);
   gaxis->SetTitleSize(0.04);
   gaxis->SetTitleColor(1);
   gaxis->SetTitleFont(62);
   gaxis->Draw();
   gaxis = new TGaxis(120,0,120,1.5,0,1.5,510,"+U");
   gaxis->SetLabelOffset(0.005);
   gaxis->SetLabelSize(0.04);
   gaxis->SetTickSize(0.03);
   gaxis->SetGridLength(0);
   gaxis->SetTitleOffset(1);
   gaxis->SetTitleSize(0.04);
   gaxis->SetTitleColor(1);
   gaxis->SetTitleFont(62);
   gaxis->Draw();
   
   TLegend *leg = new TLegend(0.10,0.62,0.45,0.75,NULL,"brNDC");
   leg->SetBorderSize(0);
   leg->SetTextSize(0.038);
   leg->SetLineColor(1);
   leg->SetLineStyle(1);
   leg->SetLineWidth(1);
   leg->SetFillColor(0);
   leg->SetFillStyle(0);
   TLegendEntry *entry=leg->AddEntry("divide_h_dxy_trg_EMTF1_by_h_dxy_EMTF1","Beam axis-constrained","pe");

   ci = TColor::GetColor("#3f90da");
   entry->SetLineColor(ci);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);

   ci = TColor::GetColor("#e42536");
   entry->SetLineColor(ci);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);

   ci = TColor::GetColor("#e42536");
   entry->SetMarkerColor(ci);
   entry->SetMarkerStyle(26);
   //entry->SetMarkerSize(2.5);
   entry->SetMarkerSize(3);
   entry->SetTextFont(42);
   entry=leg->AddEntry("divide_h_dxy_NN_trg_EMTF1_by_h_dxy_EMTF1","Beam axis-unconstrained","pe");

   ci = TColor::GetColor("#3f90da");
   entry->SetLineColor(ci);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);

   ci = TColor::GetColor("#3f90da");
   entry->SetMarkerColor(ci);
   entry->SetMarkerStyle(21);
   //entry->SetMarkerSize(2.5);
   entry->SetMarkerSize(3);
   entry->SetTextFont(42);
   leg->Draw();


   TLegend *leg2 = new TLegend(0.47,0.62,0.83,0.75,NULL,"brNDC");
   leg2->SetBorderSize(0);
   leg2->SetTextSize(0.038);
   leg2->SetLineColor(1);
   leg2->SetLineStyle(1);
   leg2->SetLineWidth(1);
   leg2->SetFillColor(0);
   leg2->SetFillStyle(0);
   TLegendEntry *entry2=leg2->AddEntry("divide_h_dxy_trg_EMTF2_by_h_dxy_EMTF2","Beam axis-constrained","pe");

   ci = TColor::GetColor("#e42536");
   entry2->SetLineColor(ci);
   entry2->SetLineStyle(1);
   entry2->SetLineWidth(1);

   ci = TColor::GetColor("#e42536");
   entry2->SetMarkerColor(ci);
   entry2->SetMarkerStyle(26);
   //entry2->SetMarkerSize(2.5);
   entry2->SetMarkerSize(3);
   entry2->SetTextFont(42);

   entry2=leg2->AddEntry("divide_h_dxy_NN_trg_EMTF2_by_h_dxy_EMTF2","Beam axis-unconstrained","pe");

   ci = TColor::GetColor("#e42536");
   entry2->SetLineColor(ci);
   entry2->SetLineStyle(1);
   entry2->SetLineWidth(1);

   ci = TColor::GetColor("#e42536");
   entry2->SetMarkerColor(ci);
   entry2->SetMarkerStyle(22);
   //entry2->SetMarkerSize(2.5);
   entry2->SetMarkerSize(3);
   entry2->SetTextFont(42);
   leg2->Draw();

   TLatex *   tex = new TLatex(0.13,0.84,"#font[42]{L1T p_{T} > 10 GeV, gen p_{T} > 15 GeV}");
   tex->SetNDC();
   tex->SetTextSize(0.04);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.13,0.77,"#font[42]{1.24 < |#eta^{gen}_{st2}| < 1.6}");
   tex->SetNDC();
   tex->SetTextSize(0.04);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.5,0.77,"#font[42]{1.6 < |#eta^{gen}_{st2}| < 2.0}");
   tex->SetNDC();
   tex->SetTextSize(0.04);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.1,0.92,"#bf{ #font[61]{CMS}} #font[52]{Simulation}");
   tex->SetNDC();
   tex->SetTextSize(0.045);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.75,0.92,"#font[42]{(13.6 TeV)}");
   tex->SetNDC();
   tex->SetTextSize(0.04);
   tex->SetLineWidth(2);
   tex->Draw();
   c->Modified();
   c->SetSelected(c);
   c->SaveAs("Figure_039-c.pdf");
}
