package com.example.mdp_14;

public class ImageIdMapper {
    public static String mapImageId(String receivedId) {
        switch (receivedId) {
            case "1": return "11";
            case "2": return "12";
            case "3": return "13";
            case "4": return "14";
            case "5": return "15";
            case "6": return "16";
            case "7": return "17";
            case "8": return "18";
            case "9": return "19";
            case "A": return "20";
            case "B": return "21";
            case "C": return "22";
            case "D": return "23";
            case "E": return "24";
            case "F": return "25";
            case "G": return "26";
            case "H": return "27";
            case "S": return "28";
            case "T": return "29";
            case "U": return "30";
            case "V": return "31";
            case "W": return "32";
            case "X": return "33";
            case "Y": return "34";
            case "Z": return "35";
            case "UP": return "36";
            case "DOWN": return "37";
            case "RIGHT": return "38";
            case "LEFT": return "39";
            case "STOP": return "40";
            default: return receivedId; // Return as-is if no mapping
        }
    }
}
